"""Tests for per-request throughput timings exposed in ``UsageInfo``.

Covers ``CompletionTimingsInfo.from_stats`` derivation and propagation of the
``timings`` field through the ``UsageInfo`` schema (streaming and non-streaming
paths assemble the same object, so schema-level coverage exercises both).
"""

from __future__ import annotations

from dataclasses import dataclass

from app.schemas.openai import CompletionTimingsInfo, UsageInfo


@dataclass
class _Stats:
    """Minimal stand-in for a GenerationResponse / BatchChunk."""

    prompt_tps: float = 0.0
    generation_tps: float = 0.0


def test_from_stats_rounds_to_two_decimals() -> None:
    """Throughput values are rounded to two decimal places."""
    timings = CompletionTimingsInfo.from_stats(_Stats(prompt_tps=123.4567, generation_tps=42.019))
    assert timings.prompt_tps == 123.46
    assert timings.generation_tps == 42.02


def test_from_stats_zero_becomes_none() -> None:
    """Zero throughput (no data reported) is surfaced as None, not 0.0."""
    timings = CompletionTimingsInfo.from_stats(_Stats(prompt_tps=0.0, generation_tps=0.0))
    assert timings.prompt_tps is None
    assert timings.generation_tps is None


def test_from_stats_missing_attributes_become_none() -> None:
    """Objects lacking the timing attributes yield None fields."""
    timings = CompletionTimingsInfo.from_stats(object())
    assert timings.prompt_tps is None
    assert timings.generation_tps is None


def test_from_stats_partial_data() -> None:
    """Only the populated field is reported; the other stays None."""
    timings = CompletionTimingsInfo.from_stats(_Stats(prompt_tps=0.0, generation_tps=55.5))
    assert timings.prompt_tps is None
    assert timings.generation_tps == 55.5


def test_usage_info_serializes_timings() -> None:
    """``timings`` round-trips through ``UsageInfo.model_dump`` (stream path)."""
    usage = UsageInfo(
        prompt_tokens=10,
        completion_tokens=20,
        total_tokens=30,
        timings=CompletionTimingsInfo.from_stats(_Stats(prompt_tps=100.0, generation_tps=50.0)),
    )
    dumped = usage.model_dump()
    assert dumped["timings"] == {"prompt_tps": 100.0, "generation_tps": 50.0}


def test_usage_info_timings_defaults_none() -> None:
    """``timings`` is omitted/None when not provided (backward compatible)."""
    usage = UsageInfo(prompt_tokens=1, completion_tokens=2, total_tokens=3)
    assert usage.timings is None
