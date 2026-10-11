# Starter RAG Project: SQuAD v2 + Chroma + Claude

A basic retrieval-augmented generation (RAG) project. It indexes Wikipedia passages from
[SQuAD v2](https://huggingface.co/datasets/rajpurkar/squad_v2) into a local
[Chroma](https://www.trychroma.com/) vector DB and answers questions with Claude, citing the passages it used.

```
INGEST: SQuAD passages → embed (bge-small, local) → Chroma
QUERY:  question → embed → top-k search → Claude (answer + [n] citations, or "I don't know")
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt && pip install -e .
cp .env.example .env   # add your ANTHROPIC_API_KEY
```

## Run

```bash
# 1. Index passages (validation split ≈ 1.2k passages, a minute or two on CPU)
python -m rag.ingest --split validation

# 2. Ask from the terminal
python -m rag.cli "What century did the Normans first gain their separate identity?"

# 3. Chat UI
streamlit run app.py
```

## Evaluate

SQuAD v2 includes gold answers and unanswerable questions, so you can measure quality:

```bash
python -m rag.evaluate --n 500              # retrieval hit@5 only (no API calls)
python -m rag.evaluate --n 50 --generate    # + answer accuracy (calls Claude)
```

Per-question results go to `eval/results.jsonl`.

## Layout

| File | Purpose |
|---|---|
| `src/rag/config.py` | Settings from `.env` (model, embedding model, top-k) |
| `src/rag/store.py` | Chroma persistent collection + embedding function |
| `src/rag/ingest.py` | Download SQuAD v2, dedupe passages, upsert to Chroma |
| `src/rag/retrieve.py` | Top-k vector search |
| `src/rag/generate.py` | Prompt construction + Claude call |
| `src/rag/cli.py` | Ask questions from the terminal |
| `src/rag/evaluate.py` | Retrieval hit@k and answer accuracy |
| `app.py` | Streamlit chat UI |

## Notes

- Default model is `claude-opus-5-5` with low effort; change `CLAUDE_MODEL` in `.env` (e.g. `claude-sonnet-5-5` or `claude-haiku-5-5` for cheaper runs).
- Server-side refusal fallback is enabled (`fallbacks="default"`), so a request declined by a safety classifier is retried on another model.

## Next steps

- Index the `train` split (~19k passages) for a bigger corpus.
- Add hybrid search (BM25 + vectors) and a re-ranker (`BAAI/bge-reranker-base`).
- Try Natural Questions (`google-research-datasets/natural_questions`). It's much larger
  (~40 GB) with long HTML pages, so you'll need real chunking.
- Swap Chroma for Qdrant or pgvector when you need a server.
