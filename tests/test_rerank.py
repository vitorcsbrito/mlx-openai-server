"""Tests for the ``rerank`` model type and the ``/v1/rerank`` endpoint.

Covers the full path: config validation, handler-type mapping, the
``MLX_Rerank`` batching wrapper, the handler's queue submission, the response
builder's sort/top_n/return_documents semantics, and the endpoint's guard rails.

The MLX model itself is stubbed — these tests pin the plumbing, not the weights.
"""

from __future__ import annotations

import importlib
import sys
import types
from typing import Any

from fastapi.responses import JSONResponse
import pytest

from app.config import VALID_MODEL_TYPES, MLXServerConfig, ModelEntryConfig
from app.core.handler_process import HandlerProcessProxy
from app.schemas.openai import RerankRequest

# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------


class _FakeScores:
    """Stand-in for the ``mx.array`` returned by ``Model.rerank``."""

    def __init__(self, values: list[float]) -> None:
        self._values = values

    def reshape(self, *_shape: int) -> _FakeScores:
        return self

    def tolist(self) -> list[float]:
        return list(self._values)


class _FakeRerankModel:
    """Reranker stub that records the payloads it was asked to score."""

    def __init__(self, scores_by_document: dict[str, float] | None = None) -> None:
        self.calls: list[dict[str, Any]] = []
        self._scores = scores_by_document or {}

    def rerank(self, payload: dict[str, Any], processor: Any) -> _FakeScores:
        self.calls.append(payload)
        assert processor is not None
        return _FakeScores([self._scores.get(d["text"], 0.5) for d in payload["documents"]])


class _NoRerankModel:
    """A loadable model that is not a cross-encoder."""


def _load_rerank_model_module(model: Any) -> Any:
    """Import ``app.models.mlx_rerank`` with ``mlx_embeddings.utils.load`` stubbed."""

    fake_utils = types.ModuleType("mlx_embeddings.utils")
    fake_utils.load = lambda _path: (model, object())

    fake_pkg = types.ModuleType("mlx_embeddings")
    fake_pkg.utils = fake_utils

    module_names = ["mlx_embeddings", "mlx_embeddings.utils", "app.models.mlx_rerank"]
    original: dict[str, types.ModuleType | None] = {
        name: sys.modules.get(name) for name in module_names
    }
    try:
        sys.modules["mlx_embeddings"] = fake_pkg
        sys.modules["mlx_embeddings.utils"] = fake_utils
        sys.modules.pop("app.models.mlx_rerank", None)
        return importlib.import_module("app.models.mlx_rerank")
    finally:
        sys.modules.pop("app.models.mlx_rerank", None)
        for name, module in original.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


def _load_endpoints_module() -> Any:
    """Import ``app.api.endpoints`` with lightweight handler stubs."""

    fake_lm_module = types.ModuleType("app.handler.mlx_lm")
    fake_lm_module.MLXLMHandler = object

    fake_vlm_module = types.ModuleType("app.handler.mlx_vlm")
    fake_vlm_module.MLXVLMHandler = object

    module_names = ["app.handler.mlx_lm", "app.handler.mlx_vlm", "app.api.endpoints"]
    original: dict[str, types.ModuleType | None] = {
        name: sys.modules.get(name) for name in module_names
    }
    try:
        sys.modules["app.handler.mlx_lm"] = fake_lm_module
        sys.modules["app.handler.mlx_vlm"] = fake_vlm_module
        sys.modules.pop("app.api.endpoints", None)
        return importlib.import_module("app.api.endpoints")
    finally:
        sys.modules.pop("app.api.endpoints", None)
        for name, module in original.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


class _StubHandler:
    """Handler stub exposing just what ``/v1/rerank`` touches."""

    def __init__(self, handler_type: str = "rerank", scores: list[float] | None = None) -> None:
        self.handler_type = handler_type
        self._scores = scores if scores is not None else [0.1, 0.9]
        self.received: list[RerankRequest] = []
        self.raises: Exception | None = None

    async def generate_rerank_response(self, request: RerankRequest) -> list[float]:
        self.received.append(request)
        if self.raises is not None:
            raise self.raises
        return self._scores


def _make_raw_request(handler: Any | None) -> Any:
    """Build a minimal request-like object backed by a single-handler registry."""

    class _Registry:
        def get_handler(self, model_id: str) -> Any:
            if handler is None:
                raise KeyError(model_id)
            return handler

        def list_model_ids(self) -> list[str]:
            return ["reranker"]

    return types.SimpleNamespace(
        app=types.SimpleNamespace(state=types.SimpleNamespace(registry=_Registry())),
        state=types.SimpleNamespace(request_id="req-test"),
    )


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


def test_rerank_is_a_valid_model_type() -> None:
    """``rerank`` is accepted by ``ModelEntryConfig`` validation."""
    assert "rerank" in VALID_MODEL_TYPES
    entry = ModelEntryConfig(model_path="some/reranker", model_type="rerank")
    assert entry.served_model_name == "some/reranker"
    assert entry.rerank_batch_size == 8


def test_rerank_batch_size_is_configurable() -> None:
    """``rerank_batch_size`` round-trips from the YAML entry."""
    entry = ModelEntryConfig(model_path="some/reranker", model_type="rerank", rerank_batch_size=32)
    assert entry.rerank_batch_size == 32


def test_rerank_batch_size_survives_single_model_cli_conversion() -> None:
    """The CLI path carries ``rerank_batch_size`` into the entry config."""
    cfg = MLXServerConfig(model_path="some/reranker", model_type="rerank", rerank_batch_size=4)
    assert cfg.to_model_entry_config().rerank_batch_size == 4


def test_invalid_model_type_still_rejected() -> None:
    """Adding ``rerank`` did not loosen validation for unknown types."""
    with pytest.raises(ValueError, match="Invalid model_type"):
        ModelEntryConfig(model_path="x", model_type="reranker")


def test_handler_process_maps_rerank_type() -> None:
    """The subprocess proxy maps the ``rerank`` model type to a ``rerank`` handler type."""
    assert HandlerProcessProxy._MODEL_TYPE_TO_HANDLER_TYPE["rerank"] == "rerank"


# ---------------------------------------------------------------------------
# MLX_Rerank wrapper
# ---------------------------------------------------------------------------


def test_rerank_model_scores_all_documents_in_order() -> None:
    """Scores come back one per document, aligned with the input order."""
    fake = _FakeRerankModel({"a": 0.2, "b": 0.8, "c": 0.4})
    module = _load_rerank_model_module(fake)

    scores = module.MLX_Rerank("stub")("q", ["a", "b", "c"])

    assert scores == [0.2, 0.8, 0.4]


def test_rerank_model_batches_documents() -> None:
    """Documents are split into ``batch_size`` chunks, preserving overall order."""
    fake = _FakeRerankModel({str(i): float(i) for i in range(5)})
    module = _load_rerank_model_module(fake)

    scores = module.MLX_Rerank("stub")("q", [str(i) for i in range(5)], batch_size=2)

    assert scores == [0.0, 1.0, 2.0, 3.0, 4.0]
    assert [len(call["documents"]) for call in fake.calls] == [2, 2, 1]


def test_rerank_model_batch_size_zero_does_not_hang() -> None:
    """A non-positive batch size is clamped to 1 rather than looping forever."""
    fake = _FakeRerankModel()
    module = _load_rerank_model_module(fake)

    scores = module.MLX_Rerank("stub")("q", ["a", "b"], batch_size=0)

    assert len(scores) == 2
    assert [len(call["documents"]) for call in fake.calls] == [1, 1]


def test_rerank_model_forwards_instruction_when_set() -> None:
    """A caller-supplied instruction reaches the payload; otherwise it is omitted."""
    fake = _FakeRerankModel()
    module = _load_rerank_model_module(fake)
    model = module.MLX_Rerank("stub")

    model("q", ["a"], instruction="Find the recipe")
    model("q", ["a"])

    assert fake.calls[0]["instruction"] == "Find the recipe"
    assert "instruction" not in fake.calls[1]


def test_rerank_model_empty_documents_skips_inference() -> None:
    """An empty candidate list returns no scores without touching the model."""
    fake = _FakeRerankModel()
    module = _load_rerank_model_module(fake)

    assert module.MLX_Rerank("stub")("q", []) == []
    assert fake.calls == []


def test_rerank_model_rejects_non_cross_encoder() -> None:
    """Loading an embedding-only model as a reranker fails loudly at startup."""
    module = _load_rerank_model_module(_NoRerankModel())

    with pytest.raises(ValueError, match="does not support reranking"):
        module.MLX_Rerank("stub")


def test_rerank_model_load_failure_is_wrapped() -> None:
    """A backend load error surfaces as a ``ValueError`` naming the cause."""
    module = _load_rerank_model_module(_FakeRerankModel())

    def _boom(_path: str) -> Any:
        raise RuntimeError("no such repo")

    module.load = _boom
    with pytest.raises(ValueError, match="no such repo"):
        module.MLX_Rerank("stub")


def test_rerank_model_cleanup_is_idempotent() -> None:
    """``cleanup()`` can run twice (explicitly, then via ``__del__``)."""
    module = _load_rerank_model_module(_FakeRerankModel())
    model = module.MLX_Rerank("stub")

    model.cleanup()
    model.cleanup()

    assert not hasattr(model, "model")


# ---------------------------------------------------------------------------
# Handler
# ---------------------------------------------------------------------------


def _load_rerank_handler_module(model: Any) -> Any:
    """Import ``app.handler.mlx_rerank`` with the model wrapper stubbed out."""

    fake_model_module = types.ModuleType("app.models.mlx_rerank")
    fake_model_module.MLX_Rerank = lambda _path: model

    original = sys.modules.get("app.models.mlx_rerank")
    try:
        sys.modules["app.models.mlx_rerank"] = fake_model_module
        sys.modules.pop("app.handler.mlx_rerank", None)
        return importlib.import_module("app.handler.mlx_rerank")
    finally:
        sys.modules.pop("app.handler.mlx_rerank", None)
        if original is None:
            sys.modules.pop("app.models.mlx_rerank", None)
        else:
            sys.modules["app.models.mlx_rerank"] = original


class _CallableModel:
    """Model stub invoked the way ``InferenceWorker`` invokes it."""

    def __init__(self) -> None:
        self.kwargs: dict[str, Any] | None = None
        self.cleaned = False

    def __call__(self, **kwargs: Any) -> list[float]:
        self.kwargs = kwargs
        return [0.3, 0.7]

    def cleanup(self) -> None:
        self.cleaned = True


@pytest.mark.asyncio
async def test_handler_submits_request_fields_to_worker() -> None:
    """The handler forwards query, documents, instruction, and batch size."""
    model = _CallableModel()
    module = _load_rerank_handler_module(model)
    handler = module.MLXRerankHandler("stub", batch_size=3)
    await handler.initialize({"queue_size": 4, "timeout": 30})

    try:
        scores = await handler.generate_rerank_response(
            RerankRequest(
                model="reranker",
                query="q",
                documents=["a", "b"],
                instruction="rank them",
            )
        )
    finally:
        await handler.cleanup()

    assert scores == [0.3, 0.7]
    assert model.kwargs == {
        "query": "q",
        "documents": ["a", "b"],
        "instruction": "rank them",
        "batch_size": 3,
    }
    assert model.cleaned is True


@pytest.mark.asyncio
async def test_handler_declares_rerank_handler_type_and_model_listing() -> None:
    """The handler advertises itself for ``/v1/models`` and endpoint dispatch."""
    module = _load_rerank_handler_module(_CallableModel())
    handler = module.MLXRerankHandler("some/reranker")

    models = await handler.get_models()

    assert handler.handler_type == "rerank"
    assert models[0]["id"] == "some/reranker"
    assert models[0]["owned_by"] == "local"


@pytest.mark.asyncio
async def test_handler_wraps_inference_failure_as_http_500() -> None:
    """A model-level exception becomes a 500 rather than escaping raw."""
    from fastapi import HTTPException

    class _Exploding(_CallableModel):
        def __call__(self, **kwargs: Any) -> list[float]:
            raise RuntimeError("metal oom")

    module = _load_rerank_handler_module(_Exploding())
    handler = module.MLXRerankHandler("stub")
    await handler.initialize({"queue_size": 4, "timeout": 30})

    try:
        with pytest.raises(HTTPException) as excinfo:
            await handler.generate_rerank_response(
                RerankRequest(model="reranker", query="q", documents=["a"])
            )
    finally:
        await handler.cleanup()

    assert excinfo.value.status_code == 500


@pytest.mark.asyncio
async def test_handler_reports_queue_stats() -> None:
    """Queue stats are exposed for ``/v1/queue/stats``."""
    module = _load_rerank_handler_module(_CallableModel())
    handler = module.MLXRerankHandler("stub")
    await handler.initialize({"queue_size": 4, "timeout": 30})

    try:
        stats = await handler.get_queue_stats()
    finally:
        await handler.cleanup()

    assert "queue_stats" in stats


# ---------------------------------------------------------------------------
# Response builder
# ---------------------------------------------------------------------------


def test_create_response_rerank_sorts_by_descending_score() -> None:
    """Results are ordered best-first while ``index`` still points at the input."""
    endpoints = _load_endpoints_module()

    response = endpoints.create_response_rerank([0.1, 0.9, 0.5], ["a", "b", "c"], "reranker")

    assert [r.index for r in response.results] == [1, 2, 0]
    assert [r.relevance_score for r in response.results] == [0.9, 0.5, 0.1]
    assert response.model == "reranker"
    assert all(r.document is None for r in response.results)


def test_create_response_rerank_applies_top_n() -> None:
    """``top_n`` truncates after sorting, keeping the highest scores."""
    endpoints = _load_endpoints_module()

    response = endpoints.create_response_rerank(
        [0.1, 0.9, 0.5], ["a", "b", "c"], "reranker", top_n=2
    )

    assert [r.index for r in response.results] == [1, 2]


def test_create_response_rerank_top_n_zero_returns_nothing() -> None:
    """``top_n=0`` is honoured rather than falling through to "all"."""
    endpoints = _load_endpoints_module()

    response = endpoints.create_response_rerank([0.1, 0.9], ["a", "b"], "reranker", top_n=0)

    assert response.results == []


def test_create_response_rerank_top_n_above_length_is_safe() -> None:
    """Asking for more results than documents returns all of them."""
    endpoints = _load_endpoints_module()

    response = endpoints.create_response_rerank([0.1, 0.9], ["a", "b"], "r", top_n=99)

    assert len(response.results) == 2


def test_create_response_rerank_echoes_documents_when_requested() -> None:
    """``return_documents`` attaches the text of the *sorted* result."""
    endpoints = _load_endpoints_module()

    response = endpoints.create_response_rerank(
        [0.1, 0.9], ["alpha", "beta"], "reranker", return_documents=True
    )

    assert response.results[0].document.text == "beta"
    assert response.results[1].document.text == "alpha"


def test_create_response_rerank_empty_input() -> None:
    """No documents yields an empty, well-formed response."""
    endpoints = _load_endpoints_module()

    response = endpoints.create_response_rerank([], [], "reranker")

    assert response.object == "list"
    assert response.results == []


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_rerank_endpoint_returns_sorted_results() -> None:
    """The happy path scores documents and returns them best-first."""
    endpoints = _load_endpoints_module()
    handler = _StubHandler(scores=[0.2, 0.8])
    request = RerankRequest(model="reranker", query="q", documents=["a", "b"])

    response = await endpoints.rerank(request, _make_raw_request(handler))

    assert [r.index for r in response.results] == [1, 0]
    assert handler.received[0].query == "q"


@pytest.mark.asyncio
async def test_rerank_endpoint_rejects_non_rerank_handler() -> None:
    """Pointing ``/v1/rerank`` at an LM model is a 400, not a crash."""
    endpoints = _load_endpoints_module()
    request = RerankRequest(model="reranker", query="q", documents=["a"])

    response = await endpoints.rerank(request, _make_raw_request(_StubHandler(handler_type="lm")))

    assert isinstance(response, JSONResponse)
    assert response.status_code == 400


@pytest.mark.asyncio
async def test_rerank_endpoint_returns_503_without_handler() -> None:
    """An unresolvable model yields 503 rather than an unhandled exception."""
    endpoints = _load_endpoints_module()
    request = RerankRequest(model="reranker", query="q", documents=["a"])

    raw_request = _make_raw_request(None)
    raw_request.app.state.registry = None
    raw_request.app.state.handler = None

    response = await endpoints.rerank(request, raw_request)

    assert isinstance(response, JSONResponse)
    assert response.status_code == 503


@pytest.mark.asyncio
async def test_rerank_endpoint_wraps_handler_failure_as_500() -> None:
    """An unexpected handler error is reported as a 500 JSON error body."""
    endpoints = _load_endpoints_module()
    handler = _StubHandler()
    handler.raises = RuntimeError("boom")
    request = RerankRequest(model="reranker", query="q", documents=["a"])

    response = await endpoints.rerank(request, _make_raw_request(handler))

    assert isinstance(response, JSONResponse)
    assert response.status_code == 500


@pytest.mark.asyncio
async def test_rerank_endpoint_propagates_http_exception() -> None:
    """An ``HTTPException`` from the handler keeps its own status code."""
    from fastapi import HTTPException

    endpoints = _load_endpoints_module()
    handler = _StubHandler()
    handler.raises = HTTPException(status_code=504, detail="timeout")
    request = RerankRequest(model="reranker", query="q", documents=["a"])

    with pytest.raises(HTTPException) as excinfo:
        await endpoints.rerank(request, _make_raw_request(handler))

    assert excinfo.value.status_code == 504


@pytest.mark.asyncio
async def test_rerank_endpoint_honours_top_n_and_return_documents() -> None:
    """Request-level shaping options reach the response."""
    endpoints = _load_endpoints_module()
    handler = _StubHandler(scores=[0.2, 0.8, 0.5])
    request = RerankRequest(
        model="reranker",
        query="q",
        documents=["a", "b", "c"],
        top_n=1,
        return_documents=True,
    )

    response = await endpoints.rerank(request, _make_raw_request(handler))

    assert len(response.results) == 1
    assert response.results[0].document.text == "b"


def test_rerank_route_is_registered() -> None:
    """``/v1/rerank`` is wired into the router."""
    endpoints = _load_endpoints_module()

    paths = {route.path for route in endpoints.router.routes}

    assert "/v1/rerank" in paths
