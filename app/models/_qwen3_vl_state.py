from typing import Any


def reset_rope_cache(model: Any) -> None:
    """Drop mlx_vlm's mRoPE cache before a one-shot forward pass.

    mlx_embeddings 0.1.0 slices `language_model._position_ids` as 3-D
    (models/qwen3_vl/model.py:92) but caches the 2-D array the text-only branch of
    `get_rope_index` returns, without mlx_vlm's rank and batch guards
    (language.py:560-577) — so rerank raises "Too many indices for array with 2
    dimensions", and reuses another batch's left padding where shapes align.
    A single forward gains nothing from the cache. TODO(upstream): file, then unpin.
    """
    language_model = getattr(model, "language_model", None)
    if language_model is None:
        return
    # SLF001: reaching into mlx_vlm's private cache is the point of this module.
    language_model._position_ids = None  # noqa: SLF001
    language_model._rope_deltas = None  # noqa: SLF001
