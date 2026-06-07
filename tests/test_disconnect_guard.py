"""Tests for the non-streaming client-disconnect guard in app.api.endpoints."""

from __future__ import annotations

import asyncio

import pytest
from starlette.requests import ClientDisconnect

from app.api.endpoints import _await_with_disconnect_guard


class _FakeRequest:
    """Minimal stand-in exposing the ``is_disconnected`` coroutine the guard uses."""

    def __init__(self, disconnected_after_polls: int) -> None:
        self._polls = 0
        self._disconnected_after_polls = disconnected_after_polls

    async def is_disconnected(self) -> bool:
        self._polls += 1
        return self._polls > self._disconnected_after_polls


@pytest.mark.asyncio
async def test_guard_returns_result_when_generation_finishes_first() -> None:
    """A connected client gets the generation result and the task is not cancelled."""

    async def work() -> str:
        return "completed"

    request = _FakeRequest(disconnected_after_polls=1000)
    result = await _await_with_disconnect_guard(request, work(), poll_interval=0.01)
    assert result == "completed"


@pytest.mark.asyncio
async def test_guard_cancels_generation_on_disconnect() -> None:
    """A client disconnect cancels the in-flight generation and raises ClientDisconnect."""
    cancelled = asyncio.Event()

    async def work() -> str:
        try:
            await asyncio.sleep(10)
        except asyncio.CancelledError:
            cancelled.set()
            raise
        return "should not happen"

    # Disconnected on the very first poll.
    request = _FakeRequest(disconnected_after_polls=0)
    with pytest.raises(ClientDisconnect):
        await _await_with_disconnect_guard(request, work(), poll_interval=0.01)

    assert cancelled.is_set(), "expected the generation task to be cancelled on disconnect"


@pytest.mark.asyncio
async def test_guard_propagates_generation_error() -> None:
    """An error from the generation propagates unchanged when the client stays connected."""

    async def work() -> str:
        raise ValueError("boom")

    request = _FakeRequest(disconnected_after_polls=1000)
    with pytest.raises(ValueError, match="boom"):
        await _await_with_disconnect_guard(request, work(), poll_interval=0.01)
