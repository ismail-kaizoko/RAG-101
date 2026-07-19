# arxiv-rag

RAG chatbot over arXiv papers on a chosen research topic (starting with LLMs (Large Language Models) models).

## Setup

```
# download packages :
pip install -r requirements.txt
```

## Status

- [x] arXiv ingestion (search, PDF download, text extraction, JSONL storage)
- [ ] Chunking
- [ ] Embedding + FAISS index
- [ ] Retrieval (dense + hybrid + reranking)
- [ ] Generation (Phi-4-mini via Ollama)
- [ ] Evaluation
- [ ] Serving (FastAPI)

## Usage

```
<!-- python scripts/ingest.py "cat:cs.LG AND abs:diffusion" --max-results 75 -->
python scripts/ingest.py "LLM" --max-results 75

```
