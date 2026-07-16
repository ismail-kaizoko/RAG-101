# arxiv-rag

RAG chatbot over arXiv papers on a chosen research topic (starting with diffusion models).

## Setup
```
uv venv .venv --python 3.11
uv pip install -e ".[dev]"
```

## Status
- [x] Repo scaffold
- [x] arXiv ingestion (search, PDF download, text extraction, JSONL storage)
- [ ] Chunking
- [ ] Embedding + FAISS index
- [ ] Retrieval (dense + hybrid + reranking)
- [ ] Generation (Phi-4-mini via Ollama)
- [ ] Evaluation
- [ ] Serving (FastAPI)

## Usage
```
python scripts/ingest.py "cat:cs.LG AND abs:diffusion" --max-results 75
```

