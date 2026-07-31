"""Regression tests for mlx_embeddings' reuse of mlx_vlm's mRoPE position cache.

mlx_embeddings 0.1.0 caches ``language_model._position_ids`` between forward passes
and slices it as 3-D, but the text-only branch of ``get_rope_index`` produces a 2-D
array — so the second rerank of a Qwen3-VL cross-encoder raised "Too many indices for
array with 2 dimensions", and the whole ``/v1/rerank`` endpoint 500'd. The stubs below
reproduce the caching behaviour without the weights: they re-populate the cache after
each forward, exactly as ``models/qwen3_vl/model.py:101`` does.
"""

from __future__ import annotations

import importlib
import sys
import types
from typing import Any

# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------


class _FakeLanguageModel:
    """Carries the two attributes mlx_vlm caches across decode steps."""

    def __init__(self) -> None:
        # 2-D, the shape the text-only branch produces and the consumer mis-slices.
        self._position_ids = [[0, 1, 2]]
        self._rope_deltas = [[0]]


class _CachingScores:
    def __init__(self, values: list[float]) -> None:
        self._values = values

    def reshape(self, *_shape: int) -> _CachingScores:
        return self

    def tolist(self) -> list[float]:
        return list(self._values)


class _VLRerankModel:
    """Qwen3-VL reranker stub: records the cache it saw, then re-populates it."""

    def __init__(self) -> None:
        self.language_model = _FakeLanguageModel()
        self.seen: list[Any] = []

    def rerank(self, payload: dict[str, Any], processor: Any) -> _CachingScores:
        assert processor is not None
        self.seen.append(self.language_model._position_ids)
        self.language_model._position_ids = [[0, 1, 2]]
        self.language_model._rope_deltas = [[0]]
        return _CachingScores([0.5] * len(payload["documents"]))


class _VLEmbeddingOutput:
    def __init__(self, text_embeds: Any) -> None:
        self.text_embeds = text_embeds


class _VLEmbeddingModel:
    """Qwen3-VL embedding stub, caching like the reranker one."""

    def __init__(self) -> None:
        self.language_model = _FakeLanguageModel()
        self.seen: list[Any] = []

    def __call__(self, input_ids: Any, attention_mask: Any = None) -> _VLEmbeddingOutput:
        del input_ids, attention_mask
        self.seen.append(self.language_model._position_ids)
        self.language_model._position_ids = [[0, 1, 2]]
        return _VLEmbeddingOutput([[0.1, 0.2]])


class _FakeTokenizer:
    def __call__(self, texts: list[str], **_kwargs: Any) -> dict[str, list[list[int]]]:
        return {
            "input_ids": [[1, 2, 3] for _ in texts],
            "attention_mask": [[1, 1, 1] for _ in texts],
        }


def _load_module(name: str, model: Any, second: Any) -> Any:
    """Import an ``app.models`` module with ``mlx_embeddings.utils.load`` stubbed."""

    fake_utils = types.ModuleType("mlx_embeddings.utils")
    fake_utils.load = lambda _path: (model, second)

    fake_pkg = types.ModuleType("mlx_embeddings")
    fake_pkg.utils = fake_utils

    module_names = ["mlx_embeddings", "mlx_embeddings.utils", name]
    original: dict[str, types.ModuleType | None] = {n: sys.modules.get(n) for n in module_names}
    try:
        sys.modules["mlx_embeddings"] = fake_pkg
        sys.modules["mlx_embeddings.utils"] = fake_utils
        sys.modules.pop(name, None)
        return importlib.import_module(name)
    finally:
        sys.modules.pop(name, None)
        for n, module in original.items():
            if module is None:
                sys.modules.pop(n, None)
            else:
                sys.modules[n] = module


# ---------------------------------------------------------------------------
# The helper
# ---------------------------------------------------------------------------


def test_reset_rope_cache_clears_both_attributes() -> None:
    from app.models._qwen3_vl_state import reset_rope_cache

    model = _VLRerankModel()
    reset_rope_cache(model)

    assert model.language_model._position_ids is None
    assert model.language_model._rope_deltas is None


def test_reset_rope_cache_ignores_text_only_models() -> None:
    """Text encoders have no ``language_model``; the helper must not raise."""
    from app.models._qwen3_vl_state import reset_rope_cache

    reset_rope_cache(object())


# ---------------------------------------------------------------------------
# Call sites
# ---------------------------------------------------------------------------


def test_rerank_clears_the_cache_before_every_batch() -> None:
    """Each batch pads independently, so a batch must never inherit stale positions."""
    fake = _VLRerankModel()
    module = _load_module("app.models.mlx_rerank", fake, object())

    scores = module.MLX_Rerank("stub")("q", [str(i) for i in range(5)], batch_size=2)

    assert len(scores) == 5
    assert fake.seen == [None, None, None]


def test_embeddings_clears_the_cache_before_every_call() -> None:
    fake = _VLEmbeddingModel()
    module = _load_module("app.models.mlx_embeddings", fake, _FakeTokenizer())

    model = module.MLX_Embeddings("stub")
    model(["one"])
    model(["two"])

    assert fake.seen == [None, None]
