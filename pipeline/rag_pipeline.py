class PipelineError(RuntimeError):
    def __init__(self, stage: str, original: Exception):
        super().__init__(f"RAG pipeline failed at stage '{stage}': {original}")
        self.stage = stage
        self.original = original


class RAGPipeline:
    def __init__(self, config, chunks, paper_titles):
        self._history: list[dict[str, str]] = []
        self._reranker = None  # lazy: only loaded if enabled AND memory permits
        try:
            from arxiv_rag.embedding.embedder import Embedder
            embedder = Embedder(config.embedding.model_name)
            index = VectorIndex.load(config.embedding.index_dir / config.chunking.method)
            dense = DenseRetriever(index, embedder)
            bm25 = BM25Retriever(chunks)
            self._hybrid = HybridRetriever(dense, bm25, k=config.retrieval.rrf_k)
        except Exception as e:
            raise PipelineError("initialization", e) from e
        self._generator = self._build_generator()

    def _maybe_load_reranker(self):
        if not self._config.reranker.enabled:
            return None
        if self._reranker is not None:
            return self._reranker
        free_mb = free_vram_mb()
        threshold = self._config.reranker.min_free_vram_mb
        if free_mb is not None and free_mb < threshold:
            logger.warning("Skipping reranker: %d MB free, need >= %d MB.", free_mb, threshold)
            return None
        try:
            from arxiv_rag.retrieval.reranker import Reranker
            self._reranker = Reranker(self._config.reranker.model_name)
        except Exception:
            logger.warning("Reranker failed to load -- continuing without it", exc_info=True)
            return None
        return self._reranker

    def ask(self, query: str) -> str:
        try:
            results = self._hybrid.search(query, top_k=..., candidate_k=...)
            candidates = [self._chunks_by_id[cid] for cid, _ in results]
        except Exception as e:
            raise PipelineError("retrieval", e) from e

        reranker = self._maybe_load_reranker()
        try:
            final_chunks = reranker.rerank(query, candidates, top_k=self._config.retrieval.top_k) \
                if reranker else candidates[:self._config.retrieval.top_k]
        except Exception as e:
            raise PipelineError("reranking", e) from e

        try:
            prompt = build_prompt(query, final_chunks, paper_titles=self._paper_titles)
            history_text = format_history(self._history)
            if history_text:
                prompt = f"Conversation so far:\n{history_text}\n\n{prompt}"
        except Exception as e:
            raise PipelineError("prompt_construction", e) from e

        try:
            answer = self._generator.generate(prompt, max_tokens=self._config.generation.max_tokens)
        except Exception as e:
            raise PipelineError("generation", e) from e

        self._history.append({"role": "user", "content": query})
        self._history.append({"role": "assistant", "content": answer})
        return answer

    def reset_conversation(self) -> None:
        self._history = []