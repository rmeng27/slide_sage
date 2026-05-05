# SlideSage: RAG-Based Conceptual Question Generator

SlideSage is a small Retrieval-Augmented Generation (RAG) app for generating conceptual practice questions from lecture slides or course notes.

The system takes uploaded course materials, chunks them, embeds the chunks, retrieves the chunks most relevant to a topic query, and asks an LLM to generate questions, answers, explanations, difficulty levels, topic tags, and source chunk IDs.

## What this project demonstrates

This project is designed as a GenAI course value-add. It demonstrates:

- document parsing
- chunking
- embedding-based retrieval
- FAISS vector search
- retrieval-augmented generation
- source-grounded question generation
- optional LLM-as-judge grounding evaluation
- a runnable Streamlit interface

## Folder structure

```text
slidesage_skill/
├── app.py                  # Streamlit app
├── rag_core.py             # RAG pipeline functions
├── requirements.txt        # Python dependencies
├── .env.example            # API key template
├── .gitignore
├── skill.md                # Project/skill specification
├── data/                   # Optional local slide files
├── outputs/                # Optional generated outputs
└── sample_outputs/         # Example outputs for GitHub/demo
```

## Supported input files

- `.html`
- `.pdf`
- `.pptx`
- `.txt`
- `.md`

HTML lecture slides usually work best because slide boundaries are easier to preserve.

## Setup

### 1. Create a virtual environment

Mac/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Add your Anthropic API key

Copy `.env.example` to `.env`:

```bash
cp .env.example .env
```

Then edit `.env`:

```text
Anthropic_API_KEY=your_actual_api_key_here
```

Do not upload `.env` to GitHub.

## Run the app

```bash
streamlit run app.py
```

Then open the local URL shown by Streamlit, usually:

```text
http://localhost:8501
```

## How to use

1. Upload lecture slides or notes.
2. Enter a topic query, such as:
   - `RAG evaluation, faithfulness, citation precision, and citation recall`
   - `agents, tools, skills, and planning loops`
   - `multimodal embeddings, CLIP, and contrastive learning`
   - `fine-tuning for classification and embedding models`
3. Choose the number of retrieved chunks.
4. Choose the number of generated questions.
5. Click **Generate conceptual questions**.
6. Review retrieved chunks and generated questions.
7. Download the full JSON output.

## Where RAG happens

The RAG pipeline is implemented in `rag_core.py`.

```text
uploaded files
→ load_file()
→ chunk_documents()
→ build_faiss_index()
→ retrieve()
→ format_context()
→ generate_questions()
```

The most important RAG functions are:

- `build_faiss_index()` embeds lecture chunks and stores them in a FAISS vector index.
- `retrieve()` embeds the user's topic query and retrieves the most relevant chunks.
- `format_context()` formats the retrieved chunks as source-labeled context.
- `generate_questions()` sends only the retrieved context to the LLM.

This means the LLM does not generate questions from the full slide deck or from general memory alone. It generates questions based on retrieved course-specific context.

## Token/cost notes

The local embedding model does not use Anthropic tokens:

```text
sentence-transformers/all-MiniLM-L6-v2
```

Anthropic tokens are used when:

- generating questions
- running optional grounding evaluation

To reduce token usage:

- lower `Retrieved chunks`
- lower `Chunk size`
- generate fewer questions
- turn off optional grounding evaluation

## Optional evaluation

The app includes an optional grounding evaluation mode. This sends the retrieved context and generated questions to the LLM and asks it to judge whether each question is supported by the retrieved source chunks.

This is useful for reporting whether the RAG system is producing grounded outputs instead of unsupported questions.

## Suggested demo script

1. Upload several GenAI lecture slides.
2. Search for `RAG evaluation and faithfulness`.
3. Show the retrieved chunks table.
4. Generate 8 questions.
5. Open one generated question and point to its source chunk IDs.
6. Turn on optional grounding evaluation and compare results.
7. Download the JSON output.

## GitHub submission notes

Recommended files to upload:

- `app.py`
- `rag_core.py`
- `requirements.txt`
- `README.md`
- `skill.md`
- `.env.example`
- `.gitignore`
- `sample_outputs/README.md`

Do not upload:

- `.env`
- private API keys
- copyrighted lecture slides unless your instructor allows it
