"""Regression tests for on-demand model release timing on streaming responses.

A streaming endpoint returns its ``StreamingResponse`` *before* the body
iterator has produced any tokens.  Previously the endpoint's ``finally`` block
released the on-demand reference immediately, dropping ``ref_count`` to zero and
starting the idle-unload timer while generation was still in progress — so when
the idle timeout was at or below the request timeout the model could be unloaded
mid-stream.  These tests pin down that the release is now deferred until the
stream is fully consumed.
"""

from __future__ import annotations

import importlib
import sys
import types
from typing import Any

from fastapi.responses import JSONResponse, StreamingResponse
import pytest


def _load_endpoints_module() -> Any:
    """Import ``app.api.endpoints`` with lightweight handler stubs."""

    fake_lm_module = types.ModuleType("app.handler.mlx_lm")
    fake_lm_module.MLXLMHandler = object

    fake_vlm_module = types.ModuleType("app.handler.mlx_vlm")
    fake_vlm_module.MLXVLMHandler = object

    module_names = [
        "app.handler.mlx_lm",
        "app.handler.mlx_vlm",
        "app.api.endpoints",
    ]
    original_modules: dict[str, types.ModuleType | None] = {
        name: sys.modules.get(name) for name in module_names
    }

    try:
        sys.modules["app.handler.mlx_lm"] = fake_lm_module
        sys.modules["app.handler.mlx_vlm"] = fake_vlm_module
        sys.modules.pop("app.api.endpoints", None)
        return importlib.import_module("app.api.endpoints")
    finally:
        sys.modules.pop("app.api.endpoints", None)
        for name, module in original_modules.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


class _RecordingRegistry:
    """Registry stub that records on-demand release calls."""

    def __init__(self) -> None:
        self.released: list[str] = []

    async def release_on_demand(self, model_id: str) -> None:
        self.released.append(model_id)


def _make_raw_request(registry: _RecordingRegistry) -> Any:
    """Build a minimal on-demand request-like object."""

    return types.SimpleNamespace(
        app=types.SimpleNamespace(state=types.SimpleNamespace(registry=registry)),
        state=types.SimpleNamespace(request_id="req-test", on_demand_model_id="model-a"),
    )


@pytest.mark.asyncio
async def test_streaming_release_deferred_until_stream_consumed() -> None:
    """Streaming on-demand release happens only after the body is fully sent."""

    endpoints = _load_endpoints_module()
    registry = _RecordingRegistry()
    raw_request = _make_raw_request(registry)

    async def _body() -> Any:
        for chunk in (b"a", b"b", b"c"):
            yield chunk

    response = StreamingResponse(_body(), media_type="text/event-stream")
    returned = endpoints._attach_on_demand_release(raw_request, response)

    # Same response object, but release is now deferred and suppressed in the
    # endpoint's ``finally`` block.
    assert returned is response
    assert raw_request.state.on_demand_release_deferred is True
    await endpoints._release_on_demand(raw_request)
    assert registry.released == []

    # Draining the stream releases exactly once, only after the last chunk.
    collected: list[bytes] = []
    async for chunk in returned.body_iterator:
        assert registry.released == [], "released before stream finished"
        collected.append(chunk)

    assert collected == [b"a", b"b", b"c"]
    assert registry.released == ["model-a"]


@pytest.mark.asyncio
async def test_streaming_release_on_client_disconnect() -> None:
    """An aborted stream still releases the on-demand reference."""

    endpoints = _load_endpoints_module()
    registry = _RecordingRegistry()
    raw_request = _make_raw_request(registry)

    async def _body() -> Any:
        yield b"a"
        yield b"b"

    response = StreamingResponse(_body(), media_type="text/event-stream")
    returned = endpoints._attach_on_demand_release(raw_request, response)

    iterator = returned.body_iterator
    assert await iterator.__anext__() == b"a"
    assert registry.released == []

    # Simulate the server tearing down the iterator mid-stream.
    await iterator.aclose()
    assert registry.released == ["model-a"]


@pytest.mark.asyncio
async def test_non_streaming_release_is_immediate() -> None:
    """Non-streaming responses are released immediately by the caller."""

    endpoints = _load_endpoints_module()
    registry = _RecordingRegistry()
    raw_request = _make_raw_request(registry)

    response = JSONResponse(content={"ok": True})
    returned = endpoints._attach_on_demand_release(raw_request, response)

    assert returned is response
    assert getattr(raw_request.state, "on_demand_release_deferred", False) is False
    await endpoints._release_on_demand(raw_request)
    assert registry.released == ["model-a"]
