"""Tests for :class:`app.core.inference_worker.InferenceWorker`."""

from __future__ import annotations

from contextlib import contextmanager
import threading
import types
from typing import Any

import pytest

from app.core import inference_worker as inference_worker_module
from app.core.inference_worker import InferenceWorker


@pytest.mark.asyncio
async def test_submit_runs_inside_worker_thread_local_stream(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Submitted work should execute inside the worker-owned MLX stream context."""
    fake_mlx = types.ModuleType("mlx")
    fake_mx = types.ModuleType("mlx.core")
    local = threading.local()
    stream_obj = object()

    fake_mx.default_device = lambda: "gpu"
    fake_mx.new_thread_local_stream = lambda device: stream_obj

    @contextmanager
    def fake_stream(stream: object) -> Any:
        local.active_stream = stream
        try:
            yield
        finally:
            local.active_stream = None

    fake_mx.stream = fake_stream
    fake_mlx.core = fake_mx
    monkeypatch.setitem(__import__("sys").modules, "mlx", fake_mlx)
    monkeypatch.setitem(__import__("sys").modules, "mlx.core", fake_mx)

    worker = InferenceWorker()
    worker.start()
    try:
        result = await worker.submit(
            lambda: local.active_stream is stream_obj and worker._stream is stream_obj
        )
    finally:
        worker.stop()

    assert result is True
    assert worker._stream is None


@pytest.mark.asyncio
async def test_submit_timeout_clears_cache_when_abandoned_work_finishes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A timed-out submit must reclaim GPU memory once the abandoned work completes."""
    cleared = threading.Event()
    monkeypatch.setattr(inference_worker_module, "_clear_mlx_cache", cleared.set)

    release = threading.Event()

    def slow_work() -> str:
        # Outlive the caller's timeout, then finish on the worker thread.
        release.wait(timeout=5.0)
        return "done"

    worker = InferenceWorker(timeout=0.05)
    worker.start()
    try:
        with pytest.raises(TimeoutError):
            await worker.submit(slow_work)
        # Let the abandoned work item run to completion.
        release.set()
        assert cleared.wait(timeout=5.0), "expected MLX cache to be cleared for abandoned work"
    finally:
        worker.stop()


@pytest.mark.asyncio
async def test_submit_clears_cache_on_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    """A submit whose work raises must trim the MLX cache and propagate the error."""
    cleared = threading.Event()
    monkeypatch.setattr(inference_worker_module, "_clear_mlx_cache", cleared.set)

    def boom() -> None:
        raise RuntimeError("kaboom")

    worker = InferenceWorker()
    worker.start()
    try:
        with pytest.raises(RuntimeError, match="kaboom"):
            await worker.submit(boom)
        assert cleared.wait(timeout=5.0), "expected MLX cache to be cleared after failure"
    finally:
        worker.stop()


@pytest.mark.asyncio
async def test_submit_stream_clears_cache_on_cancellation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Closing the stream early (client disconnect) must reclaim GPU memory."""
    cleared = threading.Event()
    monkeypatch.setattr(inference_worker_module, "_clear_mlx_cache", cleared.set)

    def endless() -> Any:
        while True:
            yield "chunk"

    worker = InferenceWorker()
    worker.start()
    try:
        stream = worker.submit_stream(endless)
        assert await stream.__anext__() == "chunk"
        # Simulate a client disconnect: closing the async generator sets the
        # worker's cancel_event, which should break the loop and clear memory.
        await stream.aclose()
        assert cleared.wait(timeout=5.0), "expected MLX cache to be cleared on cancellation"
    finally:
        worker.stop()
