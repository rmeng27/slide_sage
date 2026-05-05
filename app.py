"""
SlideSage Streamlit App

Run with:
    python -m streamlit run app.py
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st

from rag_core import (
    build_faiss_index,
    chunk_documents,
    evaluate_grounding,
    generate_questions,
    load_file,
    retrieve,
)

st.set_page_config(page_title="SlideSage", page_icon="📚", layout="wide")

st.title("📚 SlideSage: RAG-Based Conceptual Question Generator")
st.write(
    "Upload lecture slides, enter a topic, and generate conceptual practice questions "
    "grounded in the uploaded course material."
)

with st.sidebar:
    st.header("Settings")

    embedding_model_name = st.text_input(
        "Embedding model",
        value="sentence-transformers/all-MiniLM-L6-v2",
        help="Local sentence-transformer model. This does not use LLM API tokens.",
    )

    llm_model = st.selectbox(
        "LLM model",
        options=["claude-haiku-4-5-20251001"],
        index=0,
        help="Anthropic model used for question generation.",
    )

    max_words = st.slider("Chunk size", min_value=100, max_value=400, value=220, step=20)
    overlap_words = st.slider("Chunk overlap", min_value=0, max_value=100, value=35, step=5)
    run_eval = st.checkbox("Run optional grounding evaluation", value=False)

uploaded_files = st.file_uploader(
    "Upload lecture files",
    type=["html", "htm", "pdf", "pptx", "txt", "md"],
    accept_multiple_files=True,
)

topic_query = st.text_input(
    "Topic query",
    value="RAG, embeddings, agents, multimodal models, and fine-tuning",
)

col1, col2 = st.columns(2)

with col1:
    num_questions = st.slider("Number of questions", min_value=3, max_value=15, value=8)

with col2:
    top_k = st.slider("Retrieved chunks", min_value=3, max_value=12, value=6)

run_button = st.button("Generate conceptual questions", type="primary")

if run_button:
    if not uploaded_files:
        st.error("Please upload at least one lecture file.")
        st.stop()

    if not topic_query.strip():
        st.error("Please enter a topic query.")
        st.stop()

    try:
        with st.spinner("Loading files and extracting text..."):
            docs = []

            for uploaded_file in uploaded_files:
                suffix = Path(uploaded_file.name).suffix

                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    tmp.write(uploaded_file.read())
                    tmp_path = tmp.name

                loaded_docs = load_file(tmp_path)

                # Replace temp filename with original uploaded filename.
                for doc in loaded_docs:
                    doc.source_file = uploaded_file.name

                docs.extend(loaded_docs)
                os.remove(tmp_path)

            chunks = chunk_documents(
                docs,
                max_words=max_words,
                overlap_words=overlap_words,
            )

        st.success(f"Loaded {len(docs)} document units and created {len(chunks)} chunks.")

        with st.spinner("Building local FAISS retrieval index..."):
            embedding_model, index = build_faiss_index(
                chunks,
                model_name=embedding_model_name,
            )

        with st.spinner("Retrieving relevant chunks..."):
            retrieved_chunks = retrieve(
                query=topic_query,
                model=embedding_model,
                index=index,
                chunks=chunks,
                k=top_k,
            )

        st.subheader("Retrieved source chunks")

        retrieved_df = pd.DataFrame(
            [
                {
                    "rank": i + 1,
                    "source_file": c["source_file"],
                    "unit": c["page_or_slide"],
                    "title": c["title"],
                    "similarity_score": round(c["similarity_score"], 4),
                    "chunk_id": c["chunk_id"],
                }
                for i, c in enumerate(retrieved_chunks)
            ]
        )

        st.dataframe(retrieved_df, use_container_width=True)

        with st.expander("Preview retrieved text"):
            for c in retrieved_chunks:
                st.markdown(
                    f"**{c['chunk_id']}** — {c['source_file']} / unit {c['page_or_slide']}"
                )
                st.write(c["text"][:1200] + ("..." if len(c["text"]) > 1200 else ""))
                st.divider()

        with st.spinner("Generating grounded conceptual questions..."):
            questions = generate_questions(
                retrieved_chunks=retrieved_chunks,
                num_questions=num_questions,
                llm_model=llm_model,
            )

        st.subheader("Generated conceptual questions")

        question_items = None

        if isinstance(questions, dict) and isinstance(questions.get("questions"), list):
            question_items = questions["questions"]
        elif isinstance(questions, list):
            question_items = questions
        elif isinstance(questions, dict) and "raw_output" in questions:
            st.warning(
                "Questions were not returned in valid JSON format. Showing raw output instead."
            )
            st.text_area("Raw model output", questions["raw_output"], height=400)
        else:
            st.warning(
                "Questions were not returned in the expected format. Showing raw output instead."
            )
            st.json(questions)

        if question_items:
            for i, q in enumerate(question_items, start=1):
                with st.expander(f"Question {i}: {q.get('question', 'Untitled question')}"):
                    st.markdown(f"**Difficulty:** {q.get('difficulty', 'N/A')}")
                    st.markdown(f"**Question type:** {q.get('question_type', 'N/A')}")
                    st.markdown(f"**Answer:** {q.get('answer', 'N/A')}")
                    st.markdown(f"**Explanation:** {q.get('explanation', 'N/A')}")

                    concept_tags = q.get("concept_tags", [])
                    if isinstance(concept_tags, list):
                        concept_tags = ", ".join(concept_tags)

                    source_chunks = q.get("source_chunk_ids", [])
                    if isinstance(source_chunks, list):
                        source_chunks = ", ".join(source_chunks)

                    st.markdown(f"**Concept tags:** {concept_tags or 'N/A'}")
                    st.markdown(f"**Source chunks:** {source_chunks or 'N/A'}")

        eval_results = None

        if run_eval:
            with st.spinner("Evaluating grounding/faithfulness..."):
                eval_results = evaluate_grounding(
                    retrieved_chunks=retrieved_chunks,
                    questions=questions,
                    llm_model=llm_model,
                )

            st.subheader("Grounding evaluation")
            st.json(eval_results)

        output_package = {
            "topic_query": topic_query,
            "retrieved_chunks": retrieved_chunks,
            "questions": questions,
            "grounding_evaluation": eval_results,
        }

        st.download_button(
            label="Download full output as JSON",
            data=json.dumps(output_package, indent=2),
            file_name="slidesage_output.json",
            mime="application/json",
        )

    except Exception as exc:
        st.error(f"Something went wrong: {exc}")
        st.info(
            "Common fixes: check your ANTHROPIC_API_KEY, install requirements, "
            "or try a smaller/cleaner slide file."
        )