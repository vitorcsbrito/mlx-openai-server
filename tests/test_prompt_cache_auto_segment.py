"""Tests for auto-segmented prompt cache behaviour.

Covers the role-boundary computation, the batched segment builder, the
trimmable-cache segmented insert, and the non-batched multi-checkpoint
construction. The handler is instantiated via ``__new__`` so MLX-backed model
loading is bypassed and a lightweight mock model drives the logic.
"""

from __future__ import annotations

import importlib
from pathlib import Path
import sys
import threading
import types
from typing import Any
from unittest.mock import Mock

import pytest

# Deterministic role-header tokens for the fake tokenizer.
_ROLE_TOKENS = {"system": 1001, "user": 1002, "assistant": 1003, "tool": 1004}


def _install_fake_mlx_cache_module(monkeypatch: pytest.MonkeyPatch) -> None:
    """Install a lightweight ``mlx_lm.models.cache`` stub for prompt-cache imports."""
    fake_mlx_lm = types.ModuleType("mlx_lm")
    fake_models = types.ModuleType("mlx_lm.models")
    fake_cache = types.ModuleType("mlx_lm.models.cache")

    def can_trim_prompt_cache(cache: list[Any]) -> bool:
        return bool(cache)

    def trim_prompt_cache(cache: list[Any], num_tokens: int) -> int:
        # Faithful stub: report that exactly the requested number was trimmed.
        return num_tokens

    fake_cache.can_trim_prompt_cache = can_trim_prompt_cache
    fake_cache.trim_prompt_cache = trim_prompt_cache
    fake_models.cache = fake_cache
    fake_mlx_lm.models = fake_models

    monkeypatch.setitem(sys.modules, "mlx_lm", fake_mlx_lm)
    monkeypatch.setitem(sys.modules, "mlx_lm.models", fake_models)
    monkeypatch.setitem(sys.modules, "mlx_lm.models.cache", fake_cache)


def _load_handler_class(monkeypatch: pytest.MonkeyPatch) -> type[Any]:
    """Import ``MLXLMHandler`` while stubbing MLX-backed imports for CI safety."""
    repo_root = Path(__file__).resolve().parents[1]

    fake_handler_pkg = types.ModuleType("app.handler")
    fake_handler_pkg.__path__ = [str(repo_root / "app" / "handler")]
    monkeypatch.setitem(sys.modules, "app.handler", fake_handler_pkg)

    fake_mlx_lm_model = types.ModuleType("app.models.mlx_lm")

    class _FakeMLXLM:
        pass

    fake_mlx_lm_model.MLX_LM = _FakeMLXLM
    monkeypatch.setitem(sys.modules, "app.models.mlx_lm", fake_mlx_lm_model)

    _install_fake_mlx_cache_module(monkeypatch)
    sys.modules.pop("app.handler.mlx_lm", None)
    handler_module = importlib.import_module("app.handler.mlx_lm")
    handler_module = importlib.reload(handler_module)
    return handler_module.MLXLMHandler


def _encode_messages(messages: list[dict[str, Any]]) -> list[int]:
    """Encode messages so each message's tokens depend only on that message.

    Each message contributes a role-header token followed by one token per
    content character. Because the encoding is a pure function of each message,
    a prefix of messages always encodes to a prefix of the full token stream —
    exactly the property the sentinel-substitution boundary search relies on.
    """
    out: list[int] = []
    for msg in messages:
        out.append(_ROLE_TOKENS[msg["role"]])
        out.extend(ord(ch) for ch in (msg.get("content") or ""))
    return out


def _make_handler(
    monkeypatch: pytest.MonkeyPatch,
    *,
    auto_segment: bool,
    trimmable: bool,
) -> Any:
    """Build a handler with a fake tokenizing model and the given flags."""
    handler_cls = _load_handler_class(monkeypatch)
    handler = handler_cls.__new__(handler_cls)

    model = Mock()
    model.cache_is_trimmable = trimmable
    model.create_input_prompt.side_effect = lambda messages, _kwargs: messages
    model.encode_prompt.side_effect = _encode_messages

    handler.model = model
    handler.prompt_cache_auto_segment = auto_segment
    return handler


# ---------------------------------------------------------------------------
# _compute_role_boundaries
# ---------------------------------------------------------------------------


def test_compute_role_boundaries_alternating_roles(monkeypatch: pytest.MonkeyPatch) -> None:
    """Each role change yields a boundary tagged with the ending segment's role."""
    handler = _make_handler(monkeypatch, auto_segment=True, trimmable=True)
    messages = [
        {"role": "system", "content": "AB"},
        {"role": "user", "content": "CD"},
        {"role": "assistant", "content": "EF"},
    ]
    input_ids = _encode_messages(messages)

    boundaries = handler._compute_role_boundaries(messages, input_ids, {})

    # Boundary offsets fall at the content-start of each new role block.
    assert boundaries == [(4, "system"), (7, "user")]
    assert all(0 < offset < len(input_ids) for offset, _ in boundaries)


def test_compute_role_boundaries_collapses_consecutive_same_role(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A run of same-role messages collapses into a single boundary."""
    handler = _make_handler(monkeypatch, auto_segment=True, trimmable=True)
    messages = [
        {"role": "system", "content": "AB"},
        {"role": "tool", "content": "CD"},
        {"role": "tool", "content": "EF"},
        {"role": "assistant", "content": "GH"},
    ]
    input_ids = _encode_messages(messages)

    boundaries = handler._compute_role_boundaries(messages, input_ids, {})
    roles = [role for _, role in boundaries]

    # system->tool and tool->assistant change; the tool->tool repeat is skipped.
    assert roles == ["system", "tool"]


def test_compute_role_boundaries_single_message_is_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A single message has no role transition and produces no boundaries."""
    handler = _make_handler(monkeypatch, auto_segment=True, trimmable=True)
    messages = [{"role": "user", "content": "ABC"}]

    assert handler._compute_role_boundaries(messages, _encode_messages(messages), {}) == []


# ---------------------------------------------------------------------------
# _build_role_segments
# ---------------------------------------------------------------------------


def test_build_role_segments_partitions_with_trailing_assistant(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Segments partition input_ids and the trailing segment is labelled assistant."""
    handler = _make_handler(monkeypatch, auto_segment=True, trimmable=False)
    input_ids = list(range(10))
    boundaries = [(3, "system"), (7, "user")]

    segments, segment_types = handler._build_role_segments(input_ids, boundaries)

    assert segments == [[0, 1, 2], [3, 4, 5, 6], [7, 8, 9]]
    assert segment_types == ["system", "user", "assistant"]
    # Concatenation round-trips to the original prompt.
    assert [tok for seg in segments for tok in seg] == input_ids


def test_build_role_segments_none_without_boundaries(monkeypatch: pytest.MonkeyPatch) -> None:
    """No boundaries means no segmentation."""
    handler = _make_handler(monkeypatch, auto_segment=True, trimmable=False)
    assert handler._build_role_segments([1, 2, 3], []) is None


# ---------------------------------------------------------------------------
# _insert_segmented_cache (trimmable path, 3A)
# ---------------------------------------------------------------------------


def test_insert_segmented_cache_inserts_full_plus_trimmed_segments(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Trimmable cache inserts the full entry plus one trimmed entry per boundary."""
    handler = _make_handler(monkeypatch, auto_segment=True, trimmable=True)
    handler.prompt_cache = Mock()

    cache_key = list(range(10))
    cache = [Mock(name="layer0"), Mock(name="layer1")]
    boundaries = [(3, "system"), (7, "user")]

    handler._insert_segmented_cache(cache_key, cache, boundaries)

    calls = handler.prompt_cache.insert_cache.call_args_list
    # Full entry + 2 segment entries.
    assert len(calls) == 3
    # First call is the full entry (no cache_type kwarg).
    assert list(calls[0].args[0]) == cache_key
    # Segment entries use the prefix as key and the role as cache_type.
    assert list(calls[1].args[0]) == cache_key[:3]
    assert calls[1].kwargs["cache_type"] == "system"
    assert list(calls[2].args[0]) == cache_key[:7]
    assert calls[2].kwargs["cache_type"] == "user"


def test_insert_segmented_cache_nontrimmable_inserts_full_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Non-trimmable cache never produces trimmed segment entries."""
    handler = _make_handler(monkeypatch, auto_segment=True, trimmable=False)
    handler.prompt_cache = Mock()

    handler._insert_segmented_cache(list(range(10)), [Mock()], [(3, "system")])

    assert handler.prompt_cache.insert_cache.call_count == 1


def test_insert_segmented_cache_no_boundaries_inserts_full_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Disabled / empty boundaries preserve the single full-entry behaviour."""
    handler = _make_handler(monkeypatch, auto_segment=False, trimmable=True)
    handler.prompt_cache = Mock()

    handler._insert_segmented_cache(list(range(10)), [Mock()], None)

    handler.prompt_cache.insert_cache.assert_called_once()


# ---------------------------------------------------------------------------
# _build_nonbatch_checkpoints (non-trimmable path, 3B)
# ---------------------------------------------------------------------------


def test_build_nonbatch_checkpoints_trimmable_returns_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Trimmable caches need no checkpoints (prefixes derived by trimming)."""
    handler = _make_handler(monkeypatch, auto_segment=True, trimmable=True)
    messages = [
        {"role": "system", "content": "AB"},
        {"role": "user", "content": "CD"},
    ]
    input_ids = _encode_messages(messages)

    assert handler._build_nonbatch_checkpoints(messages, input_ids, input_ids, {}) == (
        None,
        None,
        None,
    )


def test_build_nonbatch_checkpoints_auto_segment_multi(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Auto-segment on a non-trimmable cache returns a multi-checkpoint list."""
    handler = _make_handler(monkeypatch, auto_segment=True, trimmable=False)
    handler.prompt_cache = Mock()
    handler._generation_lock = threading.Lock()
    messages = [
        {"role": "system", "content": "AB"},
        {"role": "user", "content": "CD"},
        {"role": "assistant", "content": "EF"},
    ]
    input_ids = _encode_messages(messages)

    position, callback, checkpoints = handler._build_nonbatch_checkpoints(
        messages, input_ids, input_ids, {}
    )

    assert position is None
    assert callback is None
    # One checkpoint per role boundary: offsets 4 and 7, cold prefix so relative
    # positions equal the absolute offsets.
    assert [pos for pos, _ in checkpoints] == [4, 7]

    # Firing a callback persists a deep-copied checkpoint under its prefix.
    checkpoints[0][1]([Mock(name="state")])
    handler.prompt_cache.insert_cache.assert_called_once()
    insert_call = handler.prompt_cache.insert_cache.call_args
    assert list(insert_call.args[0]) == input_ids[:4]
    assert insert_call.kwargs["cache_type"] == "system"


def test_build_nonbatch_checkpoints_single_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Without auto-segment, a single last-boundary checkpoint is produced."""
    handler = _make_handler(monkeypatch, auto_segment=False, trimmable=False)
    messages = [
        {"role": "system", "content": "AB"},
        {"role": "user", "content": "CD"},
    ]
    input_ids = _encode_messages(messages)

    position, callback, checkpoints = handler._build_nonbatch_checkpoints(
        messages, input_ids, input_ids, {}
    )

    assert checkpoints is None
    assert callback is not None
    assert position is not None


def test_build_nonbatch_checkpoints_skips_boundaries_within_cached_prefix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Boundaries already covered by the loaded cache cannot be re-checkpointed."""
    handler = _make_handler(monkeypatch, auto_segment=True, trimmable=False)
    messages = [
        {"role": "system", "content": "AB"},
        {"role": "user", "content": "CD"},
        {"role": "assistant", "content": "EF"},
    ]
    input_ids = _encode_messages(messages)
    # Pretend the first boundary (offset 4) is already in the cached prefix.
    rest_input_ids = input_ids[5:]

    _, _, checkpoints = handler._build_nonbatch_checkpoints(messages, input_ids, rest_input_ids, {})

    # Only the boundary at offset 7 (> cached_prefix_len 5) survives, and its
    # relative position is offset - cached_prefix_len.
    assert [pos for pos, _ in checkpoints] == [7 - 5]
