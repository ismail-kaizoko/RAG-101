# arxiv-rag

RAG chatbot over arXiv papers on a chosen research topic (starting with LLMs (Large Language Models) models).

## Setup

```
# download packages :
pip install -r requirements.txt
```

## Status

- [x] arXiv ingestion (search, PDF download, text extraction, JSONL storage)
- [x] Chunking
- [ ] Embedding + FAISS index
- [ ] Retrieval (dense + hybrid + reranking)
- [ ] Generation (Phi-4-mini via Ollama)
- [ ] Evaluation
- [ ] Serving (FastAPI)

## Usage

- ingest docs : (using arXiv API search by query)

```
python scripts/ingest.py 'cat:cs.LG AND abs:LLM' --max-results 75

# About arXiv API query syntax :
# cat: restrics to a cathegory (EX: cs.CV for Computer Vision, cs.CL for NLP )
# abs: searches within the abstract text for the given word
#  au: search papers whose authors are specefied
# all: search across all fields.
# -> puting it all together : query = "cat:cs.CV AND abs:diffusion AND abs:image" --> searchs for "image-generation papers based on diffusion"
```

- chunk the docs :

```
pyhton scripts/chunk.py --method

# supports to methods for now : naive chunking. OR section-aware chunking
```
