"""Tests for ``/v1/models`` metadata exposure via the model registry.

Covers capability derivation and the on-demand residency fields surfaced in
``ModelRegistry.list_models()``.
"""

from __future__ import annotations

import pytest

from app.core.model_registry import ModelRegistry, derive_capabilities


@pytest.mark.parametrize(
    ("model_type", "kwargs", "expected_true"),
    [
        ("lm", {}, {"text_generation", "streaming"}),
        (
            "lm",
            {"tool_call_parser": "qwen", "reasoning_parser": "qwen3"},
            {"text_generation", "streaming", "tools", "reasoning"},
        ),
        ("lm", {"enable_auto_tool_choice": True}, {"text_generation", "streaming", "tools"}),
        ("multimodal", {}, {"text_generation", "streaming", "vision"}),
        ("embeddings", {}, {"embeddings"}),
        ("rerank", {}, {"rerank"}),
        ("whisper", {}, {"audio_transcription"}),
        ("image-generation", {}, {"image_generation"}),
        ("image-edit", {}, {"image_generation"}),
    ],
)
def test_derive_capabilities(
    model_type: str, kwargs: dict[str, object], expected_true: set[str]
) -> None:
    """Capability flags reflect the model type and configured parsers."""
    caps = derive_capabilities(model_type, **kwargs)  # type: ignore[arg-type]
    true_flags = {name for name, value in caps.items() if value}
    assert true_flags == expected_true


def test_derive_capabilities_tools_reasoning_require_text() -> None:
    """``tools``/``reasoning`` are never set for non-text model types."""
    caps = derive_capabilities(
        "embeddings",
        enable_auto_tool_choice=True,
        tool_call_parser="qwen",
        reasoning_parser="qwen3",
    )
    assert caps["tools"] is False
    assert caps["reasoning"] is False


@pytest.mark.asyncio
async def test_list_models_always_on_metadata() -> None:
    """Always-on models report resident=True and active_requests=None."""
    registry = ModelRegistry()
    await registry.register_model(
        model_id="llm-main",
        handler=object(),
        model_type="lm",
        context_length=32768,
        enable_auto_tool_choice=True,
        reasoning_parser="qwen3",
    )

    (entry,) = registry.list_models()
    assert entry["id"] == "llm-main"
    meta = entry["metadata"]
    assert meta["type"] == "lm"
    assert meta["context_length"] == 32768
    assert meta["on_demand"] is False
    assert meta["resident"] is True
    assert meta["capabilities"]["tools"] is True
    assert meta["capabilities"]["reasoning"] is True
    # Transient runtime metrics belong to /v1/queue/stats, not model metadata.
    assert "active_requests" not in meta


@pytest.mark.asyncio
async def test_list_models_on_demand_not_resident() -> None:
    """On-demand models are listed before load with resident=False."""
    registry = ModelRegistry()
    await registry.register_on_demand_model(
        model_id="vlm-od",
        model_cfg_dict={
            "tool_call_parser": "qwen",
            "reasoning_parser": None,
            "enable_auto_tool_choice": False,
        },
        model_type="multimodal",
        model_path="/models/vlm",
        context_length=8192,
        queue_config={},
        idle_timeout=60,
    )

    (entry,) = registry.list_models()
    meta = entry["metadata"]
    assert meta["on_demand"] is True
    assert meta["resident"] is False
    assert meta["capabilities"]["vision"] is True
    # Derived from tool_call_parser even with auto-tool-choice disabled.
    assert meta["capabilities"]["tools"] is True
    # Transient runtime metrics belong to /v1/queue/stats, not model metadata.
    assert "active_requests" not in meta
