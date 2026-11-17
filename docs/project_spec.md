# SlideSage — Project Specification

This document summarizes the original design goals that guided SlideSage. For the current implementation and setup instructions, see the main [README](../README.md).

## Goal

Build a reusable study assistant that converts lecture slides and course notes into conceptual practice questions grounded in the uploaded source material.

## Intended Workflow

1. Parse lecture materials from HTML, PDF, PPTX, Markdown, or text.
2. Clean and split the extracted content into retrievable chunks.
3. Generate vector embeddings for each chunk.
4. Store embeddings in a FAISS index.
5. Retrieve the chunks most relevant to a user-supplied topic.
6. Generate conceptual questions using only retrieved evidence.
7. Return answers, explanations, difficulty labels, concept tags, and source chunk IDs.
8. Optionally evaluate whether the generated output is supported by the retrieved context.

## Design Priorities

- Ground generation in source material rather than general model memory.
- Preserve source traceability through chunk IDs.
- Support multiple document formats.
- Keep retrieval local with Sentence Transformers and FAISS.
- Expose retrieval results so users can inspect what context was used.
- Keep evaluation optional because LLM-as-judge adds both cost and model dependence.

## Question Types

The design supports a mix of:
- definition
- compare / contrast
- application
- misconception check
- synthesis

## Future Extensions

Potential extensions include multimodal slide extraction, retrieval benchmarking, adaptive question difficulty, and student-answer feedback.
