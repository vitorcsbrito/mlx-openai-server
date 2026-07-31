import gc

import mlx.core as mx
from mlx_embeddings.utils import load


class MLX_Rerank:
    """Cross-encoder reranker: scores (query, document) pairs in one forward pass.

    Wraps ``mlx_embeddings`` reranker models (e.g. Qwen3-VL-Reranker), which expose
    ``rerank()`` plus a processor that builds the instruct/query/document prompt.
    """

    def __init__(self, model_path: str):
        try:
            self.model, self.processor = load(model_path)
        except Exception as e:
            raise ValueError(f"Error loading model: {e!s}")

        if not hasattr(self.model, "rerank"):
            raise ValueError(
                f"Model at '{model_path}' does not support reranking "
                "(no `rerank` method); it is not a cross-encoder reranker."
            )

    def _score_batch(
        self, query: str, documents: list[str], instruction: str | None
    ) -> list[float]:
        payload: dict = {
            "query": {"text": query},
            "documents": [{"text": d} for d in documents],
        }
        if instruction:
            payload["instruction"] = instruction

        scores = None
        try:
            scores = self.model.rerank(payload, self.processor)
            mx.eval(scores)
            return [float(s) for s in scores.reshape(-1).tolist()]
        finally:
            del scores
            mx.clear_cache()
            gc.collect()

    def __call__(
        self,
        query: str,
        documents: list[str],
        instruction: str | None = None,
        batch_size: int = 8,
    ) -> list[float]:
        """Return one relevance score per document, in input order.

        Documents are scored in batches: the cross-encoder pads every pair in a
        batch to the longest one, so an unbounded batch makes peak memory track
        the single longest document across the whole request.
        """
        if not documents:
            return []

        step = max(1, batch_size)
        scores: list[float] = []
        for start in range(0, len(documents), step):
            scores.extend(self._score_batch(query, documents[start : start + step], instruction))
        return scores

    def cleanup(self):
        """Explicitly cleanup resources."""
        try:
            if hasattr(self, "model"):
                del self.model
            if hasattr(self, "processor"):
                del self.processor

            mx.clear_cache()
            gc.collect()
        except Exception:
            pass

    def __del__(self):
        self.cleanup()
