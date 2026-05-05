# SlideSage: RAG-Based Conceptual Question Generation Skill

## 1. Skill Overview

**Skill name:** `slidesage_concept_questions`  
**Purpose:** Generate grounded conceptual practice questions from lecture slides or course notes using a retrieval-augmented generation (RAG) pipeline.

This skill is designed for a GenAI course project. It accepts lecture materials, retrieves the most relevant slide chunks, generates conceptual questions, writes answers and explanations, and cites the exact slide/chunk that supports each question. The goal is not just to produce flashcards, but to build a small educational GenAI system that demonstrates document ingestion, chunking, embeddings, semantic retrieval, grounded generation, and automated evaluation.

The course connection is direct: the skill applies RAG to reduce hallucination by searching for relevant information, presenting it to the LLM as context, and generating a grounded output that cites sources. It also uses an agent/skill framing: the model is given a reusable workflow and tools for file reading, retrieval, question generation, and evaluation.

---

## 2. Problem Statement

Students often have lecture slides but still struggle to know what the “real” conceptual questions are. Slides are usually written as compressed teaching materials, not as study questions. A generic chatbot can generate practice questions, but without retrieval and citations it may hallucinate, overgeneralize, or generate questions that are not actually supported by the course materials.

**SlideSage solves this by turning course slides into grounded conceptual questions.**

For each uploaded lecture deck, it produces:

- conceptual questions;
- difficulty labels;
- topic tags;
- model answers;
- short explanations;
- supporting source chunks;
- optional self-evaluation scores.

---

## 3. Value Add

This project goes beyond a basic prompt by implementing a full RAG workflow:

1. parse slides into structured text;
2. chunk the text by slide or section;
3. embed each chunk;
4. store the embeddings in a vector database;
5. retrieve relevant chunks by topic or learning objective;
6. generate questions only from retrieved evidence;
7. evaluate the generated questions for support, usefulness, and conceptual depth.

This makes the output more trustworthy and more aligned with the actual course than a one-shot LLM question generator.

---

## 4. Intended Users

Primary users:

- students studying from lecture slides;
- instructors who want quick conceptual checks;
- teaching assistants creating review materials;
- learners who want more active recall practice.

Secondary users:

- anyone building a RAG-based study assistant;
- anyone testing embeddings or retrieval on course materials.

---

## 5. Skill Inputs

The skill should support at least one of these formats:

- `.html` lecture slides exported from Quarto, Reveal.js, or similar tools;
- `.pdf` lecture slides;
- `.pptx` lecture slides;
- `.md` notes;
- plain `.txt` notes.

Recommended MVP input:

```text
/data/slides/lecture20_rag.html
/data/slides/lecture21_rag_and_practice.html
/data/slides/lecture25_agents_and_skills.html
```

Optional user settings:

```yaml
num_questions: 10
question_types:
  - definition
  - compare_contrast
  - application
  - misconception
  - explain_why
levels:
  - easy
  - medium
  - hard
include_answers: true
include_sources: true
include_evaluation: true
```

---

## 6. Skill Outputs

The skill produces a structured markdown or JSON output.

Example markdown output:

```markdown
## Conceptual Questions: Retrieval-Augmented Generation

### Q1. Why does RAG reduce hallucination risk compared with asking an LLM directly?

**Type:** explain_why  
**Difficulty:** medium  
**Topic:** RAG / hallucinations

**Answer:** RAG reduces hallucination risk by retrieving relevant external information first, passing that information into the model as context, and requiring the answer to be grounded in that retrieved context.

**Explanation:** The model is not relying only on its parametric memory. It has access to specific course material during generation, so the answer can be checked against the source text.

**Source:** lecture20_rag.html, slide section: “The Hallucination Problem”
```

Example JSON output:

```json
{
  "question": "Why does RAG reduce hallucination risk compared with asking an LLM directly?",
  "type": "explain_why",
  "difficulty": "medium",
  "topic": "RAG / hallucinations",
  "answer": "RAG retrieves relevant external information first, passes it to the model as context, and asks the model to generate a grounded answer.",
  "explanation": "The model is no longer answering only from memory; it is constrained by source material.",
  "source": {
    "file": "lecture20_rag.html",
    "slide_title": "The Hallucination Problem",
    "chunk_id": "lecture20_slide_004"
  },
  "evaluation": {
    "faithfulness": 0.95,
    "conceptual_depth": 0.88,
    "student_usefulness": 0.91
  }
}
```

---

## 7. System Design

### 7.1 Architecture

```text
User uploads slides
        |
        v
Slide parser
        |
        v
Cleaner + metadata extractor
        |
        v
Chunker
        |
        v
Embedding model
        |
        v
Vector database
        |
        v
Retriever
        |
        v
Question generator LLM
        |
        v
Answer + explanation generator
        |
        v
Grounding evaluator
        |
        v
Markdown / JSON study set
```

### 7.2 Core Components

| Component | Purpose | Suggested implementation |
|---|---|---|
| Loader | Read HTML/PDF/PPTX/Markdown slides | `BeautifulSoup`, `pypdf`, `python-pptx`, LangChain loaders |
| Cleaner | Remove scripts, style tags, navigation text | Python regex + BeautifulSoup |
| Chunker | Split slide content into retrievable units | chunk by slide title, then token limit |
| Embedder | Convert chunks into vectors | `sentence-transformers/all-MiniLM-L6-v2`, `all-mpnet-base-v2`, or API embeddings |
| Vector store | Store and search chunks | FAISS, Chroma, LanceDB |
| Retriever | Find relevant evidence | top-k semantic retrieval, optionally hybrid retrieval |
| Generator | Create questions and answers | OpenAI, Anthropic, Gemini, or local LLM |
| Evaluator | Check grounding and quality | LLM-as-judge or Ragas-style metrics |
| Exporter | Save outputs | Markdown, JSON, CSV |

---

## 8. Recommended Repository Structure

```text
slidesage_concept_questions/
│
├── README.md
├── skill.md
├── requirements.txt
├── .env.example
│
├── data/
│   ├── slides/
│   ├── processed/
│   └── outputs/
│
├── src/
│   ├── __init__.py
│   ├── load_slides.py
│   ├── clean_text.py
│   ├── chunk_slides.py
│   ├── embed_chunks.py
│   ├── retrieve.py
│   ├── generate_questions.py
│   ├── evaluate_questions.py
│   └── export_results.py
│
├── app/
│   └── streamlit_app.py
│
└── tests/
    ├── test_chunking.py
    ├── test_retrieval.py
    └── test_output_schema.py
```

---

## 9. Minimal Python Implementation

### 9.1 Install Dependencies

```bash
pip install beautifulsoup4 sentence-transformers faiss-cpu pandas numpy pydantic python-dotenv openai streamlit
```

Optional:

```bash
pip install pypdf python-pptx ragas langchain langchain-community chromadb
```

---

### 9.2 Define the Output Schema

```python
from pydantic import BaseModel, Field
from typing import List, Optional, Dict

class SourceChunk(BaseModel):
    file: str
    slide_title: Optional[str] = None
    chunk_id: str
    text: str

class ConceptQuestion(BaseModel):
    question: str
    question_type: str
    difficulty: str
    topic: str
    answer: str
    explanation: str
    source_chunks: List[SourceChunk]
    evaluation: Optional[Dict[str, float]] = None
```

---

### 9.3 Load and Clean HTML Slides

```python
from bs4 import BeautifulSoup
from pathlib import Path
import re


def load_html_slides(path: str) -> list[dict]:
    """Load Reveal.js/Quarto HTML slides and return slide-level text chunks."""
    path_obj = Path(path)
    html = path_obj.read_text(encoding="utf-8", errors="ignore")
    soup = BeautifulSoup(html, "html.parser")

    for tag in soup(["script", "style", "nav", "footer"]):
        tag.decompose()

    sections = soup.find_all("section")
    slides = []

    for i, section in enumerate(sections):
        text = section.get_text(" ", strip=True)
        text = re.sub(r"\s+", " ", text).strip()

        if len(text) < 50:
            continue

        title_tag = section.find(["h1", "h2", "h3"])
        title = title_tag.get_text(" ", strip=True) if title_tag else f"Slide {i}"

        slides.append({
            "file": path_obj.name,
            "slide_index": i,
            "slide_title": title,
            "text": text,
            "chunk_id": f"{path_obj.stem}_slide_{i:03d}"
        })

    return slides
```

---

### 9.4 Chunk Longer Slides

```python

def chunk_text(text: str, max_words: int = 250, overlap: int = 40) -> list[str]:
    words = text.split()
    if len(words) <= max_words:
        return [text]

    chunks = []
    start = 0
    while start < len(words):
        end = start + max_words
        chunks.append(" ".join(words[start:end]))
        start = end - overlap
    return chunks


def chunk_slides(slides: list[dict], max_words: int = 250) -> list[dict]:
    all_chunks = []
    for slide in slides:
        chunks = chunk_text(slide["text"], max_words=max_words)
        for j, chunk in enumerate(chunks):
            item = slide.copy()
            item["text"] = chunk
            item["chunk_id"] = f"{slide['chunk_id']}_part_{j}"
            all_chunks.append(item)
    return all_chunks
```

---

### 9.5 Build Embeddings and Vector Index

```python
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np


def build_faiss_index(chunks: list[dict], model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
    model = SentenceTransformer(model_name)
    texts = [c["text"] for c in chunks]
    embeddings = model.encode(texts, normalize_embeddings=True)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(np.array(embeddings, dtype="float32"))

    return model, index, chunks
```

---

### 9.6 Retrieve Relevant Context

```python

def retrieve(query: str, model, index, chunks: list[dict], k: int = 5) -> list[dict]:
    query_embedding = model.encode([query], normalize_embeddings=True)
    scores, ids = index.search(np.array(query_embedding, dtype="float32"), k)

    results = []
    for score, idx in zip(scores[0], ids[0]):
        item = chunks[idx].copy()
        item["score"] = float(score)
        results.append(item)

    return results
```

---

### 9.7 Question Generation Prompt

Use a strict grounded-generation prompt so the model does not invent topics outside the retrieved material.

```python
QUESTION_GENERATION_PROMPT = """
You are SlideSage, a study assistant that writes conceptual questions from lecture slides.

Your task is to generate {num_questions} conceptual practice questions using ONLY the provided slide context.

Rules:
1. Do not use outside knowledge unless it is explicitly present in the context.
2. Each question must test conceptual understanding, not memorization only.
3. Include a mix of question types: definition, compare/contrast, application, misconception, and explain-why.
4. Each question must include:
   - question
   - question_type
   - difficulty: easy, medium, or hard
   - topic
   - answer
   - explanation
   - supporting chunk IDs
5. If the context is not enough to support a question, do not write that question.
6. Return valid JSON only.

Slide context:
{context}
"""
```

---

### 9.8 Generate Questions with an LLM

```python
import os
import json
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def format_context(chunks: list[dict]) -> str:
    blocks = []
    for c in chunks:
        blocks.append(
            f"[chunk_id: {c['chunk_id']} | file: {c['file']} | title: {c['slide_title']}]\n{c['text']}"
        )
    return "\n\n".join(blocks)


def generate_questions(topic_query: str, retrieved_chunks: list[dict], num_questions: int = 8) -> list[dict]:
    context = format_context(retrieved_chunks)
    prompt = QUESTION_GENERATION_PROMPT.format(
        num_questions=num_questions,
        context=context
    )

    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {"role": "system", "content": "You produce valid JSON study materials grounded in source text."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3
    )

    content = response.choices[0].message.content
    return json.loads(content)
```

---

### 9.9 Grounding Evaluation Prompt

```python
EVALUATION_PROMPT = """
You are evaluating whether a generated study question is grounded in the provided lecture context.

Evaluate the item on a 0 to 1 scale for each criterion:

1. faithfulness: Are the answer and explanation fully supported by the context?
2. conceptual_depth: Does the question test understanding rather than surface memorization?
3. student_usefulness: Would this help a student prepare for an exam or discussion?
4. citation_quality: Do the cited chunk IDs actually support the answer?

Return valid JSON only:
{
  "faithfulness": number,
  "conceptual_depth": number,
  "student_usefulness": number,
  "citation_quality": number,
  "notes": "brief explanation"
}

Question item:
{question_item}

Context:
{context}
"""
```

```python

def evaluate_question(question_item: dict, source_chunks: list[dict]) -> dict:
    context = format_context(source_chunks)
    prompt = EVALUATION_PROMPT.format(
        question_item=json.dumps(question_item, indent=2),
        context=context
    )

    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {"role": "system", "content": "You are a strict evaluator of grounded educational content."},
            {"role": "user", "content": prompt}
        ],
        temperature=0
    )

    return json.loads(response.choices[0].message.content)
```

---

### 9.10 Export to Markdown

```python

def export_questions_to_markdown(questions: list[dict], output_path: str):
    lines = ["# SlideSage Conceptual Question Set\n"]

    for i, q in enumerate(questions, start=1):
        lines.append(f"## Q{i}. {q['question']}\n")
        lines.append(f"**Type:** {q.get('question_type', 'NA')}  ")
        lines.append(f"**Difficulty:** {q.get('difficulty', 'NA')}  ")
        lines.append(f"**Topic:** {q.get('topic', 'NA')}\n")
        lines.append(f"**Answer:** {q.get('answer', '')}\n")
        lines.append(f"**Explanation:** {q.get('explanation', '')}\n")

        if "supporting_chunk_ids" in q:
            lines.append("**Sources:** " + ", ".join(q["supporting_chunk_ids"]) + "\n")

        if "evaluation" in q and q["evaluation"]:
            lines.append("**Evaluation:**\n")
            for key, value in q["evaluation"].items():
                lines.append(f"- {key}: {value}")
            lines.append("")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
```

---

## 10. End-to-End Runner

```python
from pathlib import Path


def run_slidesage(slide_paths: list[str], topic_query: str, num_questions: int = 10):
    all_slides = []
    for path in slide_paths:
        if path.endswith(".html"):
            all_slides.extend(load_html_slides(path))
        else:
            raise ValueError(f"Unsupported file type for MVP: {path}")

    chunks = chunk_slides(all_slides)
    model, index, chunks = build_faiss_index(chunks)
    retrieved = retrieve(topic_query, model, index, chunks, k=8)

    questions = generate_questions(
        topic_query=topic_query,
        retrieved_chunks=retrieved,
        num_questions=num_questions
    )

    # Optional: evaluate each question against retrieved context.
    evaluated_questions = []
    for q in questions:
        q["evaluation"] = evaluate_question(q, retrieved)
        evaluated_questions.append(q)

    output_path = "data/outputs/slidesage_questions.md"
    Path("data/outputs").mkdir(parents=True, exist_ok=True)
    export_questions_to_markdown(evaluated_questions, output_path)

    return evaluated_questions
```

Example usage:

```python
slide_paths = [
    "data/slides/lecture20_rag.html",
    "data/slides/lecture21_rag_and_practice.html",
    "data/slides/lecture25_agents_and_skills.html"
]

questions = run_slidesage(
    slide_paths=slide_paths,
    topic_query="retrieval augmented generation, hallucination, agentic RAG, and RAG evaluation",
    num_questions=10
)
```

---

## 11. Streamlit Demo App

```python
import streamlit as st
from pathlib import Path

st.set_page_config(page_title="SlideSage", page_icon="📚")

st.title("📚 SlideSage")
st.subheader("RAG-Based Conceptual Question Generator")

uploaded_files = st.file_uploader(
    "Upload lecture slides",
    type=["html"],
    accept_multiple_files=True
)

topic_query = st.text_input(
    "What topic should the questions focus on?",
    value="retrieval augmented generation and hallucination"
)

num_questions = st.slider("Number of questions", 3, 20, 10)

if st.button("Generate Conceptual Questions"):
    if not uploaded_files:
        st.warning("Please upload at least one HTML slide file.")
    else:
        slide_dir = Path("data/slides")
        slide_dir.mkdir(parents=True, exist_ok=True)

        paths = []
        for file in uploaded_files:
            path = slide_dir / file.name
            path.write_bytes(file.read())
            paths.append(str(path))

        with st.spinner("Building RAG index and generating questions..."):
            questions = run_slidesage(paths, topic_query, num_questions)

        st.success("Done!")

        for i, q in enumerate(questions, start=1):
            with st.expander(f"Q{i}. {q['question']}"):
                st.write(f"**Type:** {q.get('question_type')}")
                st.write(f"**Difficulty:** {q.get('difficulty')}")
                st.write(f"**Topic:** {q.get('topic')}")
                st.write(f"**Answer:** {q.get('answer')}")
                st.write(f"**Explanation:** {q.get('explanation')}")
                st.write(f"**Sources:** {q.get('supporting_chunk_ids')}")
                if q.get("evaluation"):
                    st.json(q["evaluation"])
```

---

## 12. Skill Instruction File Version

If the project needs to be packaged as a reusable agent skill, the instruction file can be written like this:

```markdown
# slidesage_concept_questions

Use this skill when the user wants to generate study questions, conceptual checks, quiz items, or exam review material from lecture slides or course notes.

## Workflow

1. Inspect the uploaded course material.
2. Extract slide titles, bullet points, equations, code blocks, definitions, and examples.
3. Split the material into slide-level or section-level chunks.
4. Retrieve the chunks most relevant to the requested topic.
5. Generate conceptual questions using only retrieved chunks.
6. For each question, include an answer, explanation, difficulty label, topic tag, and source chunk.
7. Reject questions that are not directly supported by the source material.
8. Optionally run an evaluation pass for faithfulness, usefulness, and conceptual depth.

## Output Format

Return markdown by default. Each question should include:

- Question
- Question type
- Difficulty
- Topic
- Answer
- Explanation
- Source slide/chunk
- Evaluation scores, if requested

## Constraints

- Do not invent course content.
- Do not generate questions from outside knowledge unless the user explicitly asks for outside enrichment.
- Prefer conceptual understanding over memorization.
- Use the source chunks as evidence.
- If the slides do not contain enough information, say so.
```

---

## 13. Question Type Taxonomy

Use this taxonomy to make the output more varied and more educational.

| Question type | Purpose | Example starter |
|---|---|---|
| Definition | Check core vocabulary | “What is dense retrieval?” |
| Compare/contrast | Distinguish related ideas | “How is RAG different from fine-tuning?” |
| Explain-why | Test causal reasoning | “Why does RAG help reduce hallucination?” |
| Application | Apply concept to new case | “If a student uploads lecture slides, where would chunking happen?” |
| Misconception | Identify false assumptions | “Why is it wrong to say RAG guarantees truth?” |
| Pipeline reasoning | Understand system design | “What happens between document loading and generation?” |
| Evaluation | Judge system quality | “Why does faithfulness matter for RAG outputs?” |

---

## 14. Evaluation Plan

The project should evaluate both retrieval quality and generated-question quality.

### 14.1 Retrieval Evaluation

Create a small test set manually:

```csv
query,expected_file,expected_topic
"Why does RAG reduce hallucinations?",lecture20_rag.html,hallucination_problem
"What is citation precision?",lecture21_rag_and_practice.html,rag_evaluation
"What makes an agent different from a chatbot?",lecture25_agents_and_skills.html,agents
```

Metrics:

- **Top-1 accuracy:** Did the correct chunk appear first?
- **Top-3 accuracy:** Did the correct chunk appear in the first 3 results?
- **Mean reciprocal rank:** How high was the first correct result?
- **Mean average precision:** How well are relevant chunks ranked overall?

### 14.2 Generation Evaluation

For each generated question, score:

| Metric | Meaning |
|---|---|
| Faithfulness | Is the answer supported by the retrieved context? |
| Citation quality | Do cited chunks actually support the question and answer? |
| Conceptual depth | Does the question require understanding? |
| Student usefulness | Would this help someone study? |
| Diversity | Are question types varied? |

### 14.3 Human Evaluation Rubric

A human reviewer can score each item from 1 to 5:

| Score | Interpretation |
|---|---|
| 1 | unsupported, confusing, or useless |
| 2 | partially related but weak |
| 3 | acceptable but basic |
| 4 | useful and mostly grounded |
| 5 | highly useful, grounded, and conceptually strong |

---

## 15. Demo Plan

A good class demo should be short and concrete.

1. Upload three lecture slide files.
2. Type topic query: `RAG and evaluation`.
3. Show retrieved chunks.
4. Generate 8 conceptual questions.
5. Open one generated item and show the answer, explanation, and source chunk.
6. Show evaluation scores.
7. Explain one failure case, such as a vague question or weak citation.
8. Conclude with how RAG made the tool more grounded than a generic chatbot.

---

## 16. Extensions

### Extension A: Multimodal Slide Support

Some slides contain diagrams. A stronger version could extract slide screenshots and use a multimodal model to summarize diagrams before adding them to the retrieval index.

Pipeline:

```text
Slide image -> vision model summary -> text chunk -> embedding -> retrieval -> grounded question
```

This connects to multimodal embedding ideas where text and images can be represented in compatible vector spaces.

### Extension B: Adaptive Difficulty

Use the model to generate easy, medium, and hard versions of the same concept.

Example:

- Easy: “What does RAG stand for?”
- Medium: “Why does RAG reduce hallucination risk?”
- Hard: “What are the limitations of RAG if retrieval quality is poor?”

### Extension C: Student Feedback Loop

Let the student answer a generated question, then have the model evaluate the answer against the source chunk.

Output:

- correctness score;
- missing concept;
- improved answer;
- source reference.

### Extension D: Embedding Model Comparison

Compare retrieval results across:

- TF-IDF;
- BM25;
- `all-MiniLM-L6-v2`;
- `all-mpnet-base-v2`;
- API embeddings.

Report top-k accuracy and MAP.

---

## 17. Limitations

- The skill is only as good as the slide extraction and chunking.
- If the slides are vague, the generated questions may also be vague.
- RAG reduces hallucination risk but does not eliminate it.
- LLM-as-judge evaluation is useful but not a substitute for human review.
- Diagrams and screenshots require multimodal processing; pure text extraction may miss important visual content.

---

## 18. Project Deliverables

Recommended final deliverables for the GenAI value-add assignment:

1. `skill.md`: reusable skill instructions and workflow.
2. `README.md`: project overview and how to run.
3. `src/`: Python implementation.
4. `app/streamlit_app.py`: interactive demo.
5. `data/outputs/slidesage_questions.md`: generated sample output.
6. `evaluation_report.md`: retrieval and generation evaluation results.
7. short presentation or demo video.

---

## 19. References

Course slides used as project grounding:

- Lecture 20: RAG
- Lecture 21: RAG and Practice
- Lecture 22: Multimodal Models
- Lecture 23: Creating Embedding Models
- Lecture 24: Fine-Tuning Classification
- Lecture 25: Agents and Skills

Outside references:

- Anthropic. “Building Effective Agents.” https://www.anthropic.com/research/building-effective-agents
- LangChain Docs. “Retrieval.” https://docs.langchain.com/oss/python/langchain/retrieval
- Ragas Docs. “Available Metrics.” https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/
- Ragas Docs. “Faithfulness.” https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/faithfulness/

---

## 20. One-Sentence Project Pitch

SlideSage is a RAG-based educational skill that turns lecture slides into grounded conceptual question sets with answers, explanations, citations, and evaluation scores, showing how retrieval, embeddings, agents, and LLM-as-judge evaluation can be combined into a practical GenAI study tool.
