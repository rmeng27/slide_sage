"""
SlideSage RAG Core

This module contains the main Retrieval-Augmented Generation workflow:
1. load lecture files
2. chunk lecture content
3. embed chunks
4. retrieve relevant chunks for a topic query
5. generate conceptual questions grounded in retrieved chunks
6. optionally evaluate whether each generated question is supported by sources
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import faiss
import numpy as np
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from anthropic import Anthropic
from pypdf import PdfReader
from pptx import Presentation
from sentence_transformers import SentenceTransformer

load_dotenv()


@dataclass
class SlideDocument:
    """A slide/page/text unit before chunking."""

    source_file: str
    page_or_slide: int
    title: str
    text: str


@dataclass
class Chunk:
    """A searchable text chunk used by the RAG index."""

    chunk_id: str
    source_file: str
    page_or_slide: int
    title: str
    text: str


def clean_text(text: str) -> str:
    """Normalize whitespace and remove duplicated empty space."""
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def load_html_file(path: str | Path) -> list[SlideDocument]:
    """Load HTML slides and extract text from likely slide containers."""
    path = Path(path)
    html = path.read_text(encoding="utf-8", errors="ignore")
    soup = BeautifulSoup(html, "html.parser")

    # Try common slide containers first, then fall back to body text.
    candidates = soup.find_all(["section", "article"])
    if not candidates:
        candidates = soup.find_all("div")

    docs: list[SlideDocument] = []
    seen: set[str] = set()

    for idx, element in enumerate(candidates, start=1):
        text = clean_text(element.get_text(separator=" ", strip=True))
        if len(text.split()) < 25:
            continue

        # Avoid storing huge duplicated parent divs if the HTML has nested sections.
        fingerprint = text[:500]
        if fingerprint in seen:
            continue
        seen.add(fingerprint)

        title_tag = element.find(["h1", "h2", "h3"])
        title = clean_text(title_tag.get_text(" ", strip=True)) if title_tag else f"Slide {idx}"

        docs.append(
            SlideDocument(
                source_file=path.name,
                page_or_slide=idx,
                title=title,
                text=text,
            )
        )

    if not docs:
        body_text = clean_text(soup.get_text(separator=" ", strip=True))
        if body_text:
            docs.append(SlideDocument(path.name, 1, path.stem, body_text))

    return docs


def load_pdf_file(path: str | Path) -> list[SlideDocument]:
    """Load a PDF file page by page."""
    path = Path(path)
    reader = PdfReader(str(path))
    docs: list[SlideDocument] = []

    for idx, page in enumerate(reader.pages, start=1):
        text = clean_text(page.extract_text() or "")
        if len(text.split()) < 20:
            continue
        docs.append(SlideDocument(path.name, idx, f"Page {idx}", text))

    return docs


def load_pptx_file(path: str | Path) -> list[SlideDocument]:
    """Load a PowerPoint file slide by slide."""
    path = Path(path)
    prs = Presentation(str(path))
    docs: list[SlideDocument] = []

    for idx, slide in enumerate(prs.slides, start=1):
        texts: list[str] = []
        for shape in slide.shapes:
            if hasattr(shape, "text") and shape.text:
                texts.append(shape.text)
        text = clean_text(" ".join(texts))
        if len(text.split()) < 15:
            continue
        title = texts[0].strip().split("\n")[0] if texts else f"Slide {idx}"
        docs.append(SlideDocument(path.name, idx, clean_text(title), text))

    return docs


def load_text_file(path: str | Path) -> list[SlideDocument]:
    """Load a plain text or markdown file."""
    path = Path(path)
    text = clean_text(path.read_text(encoding="utf-8", errors="ignore"))
    if not text:
        return []
    return [SlideDocument(path.name, 1, path.stem, text)]


def load_file(path: str | Path) -> list[SlideDocument]:
    """Load one supported file type."""
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix == ".html" or suffix == ".htm":
        return load_html_file(path)
    if suffix == ".pdf":
        return load_pdf_file(path)
    if suffix == ".pptx":
        return load_pptx_file(path)
    if suffix in {".txt", ".md"}:
        return load_text_file(path)

    raise ValueError(f"Unsupported file type: {suffix}")


def chunk_documents(docs: list[SlideDocument], max_words: int = 220, overlap_words: int = 35) -> list[Chunk]:
    """Split documents into overlapping chunks for retrieval."""
    chunks: list[Chunk] = []
    step = max(1, max_words - overlap_words)

    for doc in docs:
        words = doc.text.split()
        for start in range(0, len(words), step):
            chunk_words = words[start : start + max_words]
            if len(chunk_words) < 25:
                continue

            chunk_id = f"{doc.source_file}_unit{doc.page_or_slide}_chunk{len(chunks) + 1}"
            chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    source_file=doc.source_file,
                    page_or_slide=doc.page_or_slide,
                    title=doc.title,
                    text=" ".join(chunk_words),
                )
            )

    return chunks


def build_faiss_index(
    chunks: list[Chunk],
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
) -> tuple[SentenceTransformer, faiss.IndexFlatIP]:
    """Embed chunks and store them in a FAISS inner-product index.

    Because embeddings are normalized, inner product works like cosine similarity.
    """
    if not chunks:
        raise ValueError("No chunks were provided. Check whether your files contain extractable text.")

    model = SentenceTransformer(model_name)
    texts = [chunk.text for chunk in chunks]
    embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    embeddings = np.asarray(embeddings, dtype="float32")

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    return model, index


def retrieve(
    query: str,
    model: SentenceTransformer,
    index: faiss.IndexFlatIP,
    chunks: list[Chunk],
    k: int = 6,
) -> list[dict[str, Any]]:
    """Retrieve the top-k chunks most relevant to a topic query."""
    if not query.strip():
        raise ValueError("Query cannot be empty.")

    k = min(k, len(chunks))
    query_embedding = model.encode([query], normalize_embeddings=True, show_progress_bar=False)
    query_embedding = np.asarray(query_embedding, dtype="float32")
    scores, ids = index.search(query_embedding, k)

    results: list[dict[str, Any]] = []
    for score, idx in zip(scores[0], ids[0]):
        chunk = chunks[int(idx)]
        results.append(
            {
                "chunk_id": chunk.chunk_id,
                "source_file": chunk.source_file,
                "page_or_slide": chunk.page_or_slide,
                "title": chunk.title,
                "text": chunk.text,
                "similarity_score": float(score),
            }
        )

    return results


def format_context(retrieved_chunks: list[dict[str, Any]]) -> str:
    """Format retrieved chunks as source-labeled context for the LLM."""
    blocks: list[str] = []
    for chunk in retrieved_chunks:
        blocks.append(
            f"[chunk_id: {chunk['chunk_id']} | file: {chunk['source_file']} | "
            f"unit: {chunk['page_or_slide']} | title: {chunk['title']}]\n{chunk['text']}"
        )
    return "\n\n".join(blocks)


QUESTION_GENERATION_PROMPT = """
You are an expert teaching assistant for a graduate-level Generative AI course.

Your task is to generate conceptual practice questions using ONLY the provided slide context.
Do not use outside knowledge. If the provided context is insufficient, write a question only about what is supported by the context.
Do not invent source references.

Generate {num_questions} questions.

Requirements:
- Questions should test conceptual understanding, not shallow memorization.
- Include a mix of easy, medium, and hard questions.
- Include at least one compare/contrast question.
- Include at least one application question.
- Include at least one misconception-check question.
- Answers must be concise but complete.
- Explanations must say why the answer is correct.
- Every question must cite at least one source_chunk_id from the context.

Return valid JSON only in this schema:
{{
  "questions": [
    {{
      "question": "...",
      "answer": "...",
      "explanation": "...",
      "difficulty": "easy | medium | hard",
      "question_type": "definition | compare_contrast | application | misconception_check | synthesis",
      "concept_tags": ["..."],
      "source_chunk_ids": ["..."]
    }}
  ]
}}

Slide context:
{context}
"""


def get_anthropic_client() -> Anthropic:
    """Create an Anthropic client after checking that the API key exists."""
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY is missing. Create a .env file based on .env.example, "
            "or export ANTHROPIC_API_KEY in your terminal."
        )
    return Anthropic(api_key=api_key)

def parse_json_output(content: str) -> dict[str, Any]:
    """
    Parse model output into JSON, even if the model wraps it in markdown fences.
    """
    content = content.strip()

    # Remove markdown code fences if present.
    if content.startswith("```"):
        content = re.sub(r"^```(?:json)?", "", content.strip(), flags=re.IGNORECASE)
        content = re.sub(r"```$", "", content.strip())

    # Try direct JSON parsing first.
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        # Fallback: extract the first JSON-looking object.
        match = re.search(r"\{.*\}", content, flags=re.DOTALL)
        if match:
            try:
                parsed = json.loads(match.group(0))
            except json.JSONDecodeError:
                return {"error": "Model did not return valid JSON.", "raw_output": content}
        else:
            return {"error": "Model did not return valid JSON.", "raw_output": content}

    # Normalize list output into expected dict format.
    if isinstance(parsed, list):
        return {"questions": parsed}

    # Normalize dict output.
    if isinstance(parsed, dict):
        if "questions" in parsed:
            return parsed
        if "question" in parsed:
            return {"questions": [parsed]}
        return parsed

    return {"error": "Model returned JSON, but not in a usable format.", "raw_output": parsed}

def generate_questions(
    retrieved_chunks: list[dict[str, Any]],
    num_questions: int = 8,
    llm_model: str = "claude-haiku-4-5-20251001",
) -> dict[str, Any]:
    """Generate conceptual questions grounded in retrieved slide chunks."""
    client = get_anthropic_client()
    context = format_context(retrieved_chunks)
    prompt = QUESTION_GENERATION_PROMPT.format(num_questions=num_questions, context=context)

    response = client.messages.create(
        model=llm_model,
        max_tokens=4000,
        temperature=0.3,
        system="You generate valid JSON study materials grounded only in provided source text.",
        messages=[
            {"role": "user", "content": prompt},
        ],
    )

    content = response.content[0].text if response.content else ""

    return parse_json_output(content)

FAITHFULNESS_PROMPT = """
You are evaluating whether generated study questions are grounded in the retrieved source context.

For each question, judge:
- supported: true/false
- support_score: integer from 1 to 5
- issue: brief note if unsupported or partially supported

Return valid JSON only.

Retrieved context:
{context}

Generated questions:
{questions_json}
"""


def evaluate_grounding(
    retrieved_chunks: list[dict[str, Any]],
    questions: dict[str, Any],
    llm_model: str = "claude-haiku-4-5-20251001",
) -> dict[str, Any]:
    """Optional LLM-as-judge evaluation for source grounding."""
    client = get_anthropic_client()
    context = format_context(retrieved_chunks)
    prompt = FAITHFULNESS_PROMPT.format(
        context=context,
        questions_json=json.dumps(questions, indent=2),
    )

    response = client.messages.create(
        model=llm_model,
        max_tokens=2000,
        temperature=0.0,
        system="You evaluate RAG outputs and return valid JSON only.",
        messages=[
            {"role": "user", "content": prompt},
        ],
    )

    content = response.content[0].text if response.content else ""

    try:
        return json.loads(content)
    except json.JSONDecodeError:
        return {"error": "Evaluator did not return valid JSON.", "raw_output": content}