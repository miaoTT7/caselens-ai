# Retrieval evaluation

This directory contains the repeatable CaseLens retrieval baseline. PDF files
under `pdfs/` are local inputs and remain ignored by Git.

From `Backend-ai`, run:

```bash
source .venv/bin/activate
python evaluation/run_evaluation.py
```

The runner creates or reuses the `insurance-evaluation` knowledge base,
ingests missing PDFs, runs every query at Top-5, and writes JSON and Markdown
reports under `evaluation/results/`.
