# SlideSage

<p align="center">
  <strong>A document-grounded RAG application that turns lecture materials into conceptual practice questions with answers, explanations, citations, and optional grounding evaluation.</strong>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.x-3776AB?logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/Streamlit-App-FF4B4B?logo=streamlit&logoColor=white" alt="Streamlit">
  <img src="https://img.shields.io/badge/FAISS-Vector%20Search-4B8BBE" alt="FAISS">
  <img src="https://img.shields.io/badge/RAG-Grounded%20Generation-6C63FF" alt="RAG">
</p>

## Overview

**SlideSage** helps students turn lecture slides and course notes into source-grounded study material.

The application parses uploaded documents, splits them into retrievable chunks, creates local embeddings, retrieves the chunks most relevant to a topic, and asks Anthropic Claude to generate conceptual questions using only that retrieved context.

Each generated item can include an answer, explanation, difficulty label, concept tags, and source chunk IDs. An optional evaluator checks whether generated questions are supported by the retrieved evidence.

## How It Works

```text
Uploaded documents
  → text extraction
  → chunking
  → Sentence Transformer embeddings
  → FAISS vector index
  → top-k semantic retrieval
  → Claude question generation
  → source-linked output
  → optional grounding evaluation
```

## Supported Inputs

- HTML lecture slides
- PDF files
- PowerPoint presentations
- Markdown notes
- plain-text notes

## Key Features

### Multi-format document ingestion
SlideSage extracts content from HTML, PDF, PPTX, Markdown, and text files.

### Semantic retrieval
Document chunks are embedded with `sentence-transformers/all-MiniLM-L6-v2` and indexed in FAISS for top-k retrieval.

### Grounded question generation
Only retrieved source context is passed into the generation prompt, reducing reliance on unsupported model memory.

### Source traceability
Generated questions include source chunk IDs so users can trace the output back to retrieved material.

### Optional grounding evaluation
An LLM-as-judge step can evaluate whether generated study questions are supported by the retrieved evidence.

### Structured export
The full result package—including retrieved chunks, generated questions, and optional evaluation—is downloadable as JSON.

## Tech Stack

| Layer | Tools |
|---|---|
| Application | Python, Streamlit |
| Document parsing | BeautifulSoup, pypdf, python-pptx |
| Embeddings | Sentence Transformers |
| Vector search | FAISS |
| Generation | Anthropic Claude API |
| Data handling | pandas, NumPy |

## Repository Structure

```text
slide_sage/
├── app.py
├── rag_core.py
├── requirements.txt
├── README.md
├── .env.example
├── .gitignore
├── data/
│   └── README.md
└── docs/
    └── project_spec.md
```

## Run Locally

### 1. Create a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate
# Windows: .venv\Scripts\activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure the Anthropic API key

```bash
cp .env.example .env
```

Then edit `.env`:

```text
ANTHROPIC_API_KEY=your_anthropic_api_key_here
```

The real `.env` file is excluded from version control.

### 4. Launch

```bash
streamlit run app.py
```

## Example Workflow

1. Upload one or more lecture files.
2. Enter a topic query.
3. Choose chunk size and retrieval depth.
4. Inspect the retrieved source chunks and similarity scores.
5. Generate conceptual questions.
6. Optionally run grounding evaluation.
7. Download the full output as JSON.

## Evaluation Notes

The optional evaluator checks generated questions against the retrieved source context and returns structured grounding judgments. This is a lightweight reliability check rather than a substitute for human review.

## Limitations

- Retrieval quality depends on document extraction and chunking quality.
- Image-heavy slides may lose information during text-only extraction.
- RAG reduces unsupported generation risk but does not eliminate it.
- LLM-based evaluation is itself model-dependent.
- The app requires an Anthropic API key for generation and evaluation.

## Additional Documentation

The original design specification is preserved in [`docs/project_spec.md`](docs/project_spec.md).

---

<sub>Originally developed as a graduate Generative AI project and refined for portfolio presentation.</sub>
