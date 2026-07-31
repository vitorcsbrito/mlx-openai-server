import gc
from http import HTTPStatus
import time
from typing import Any

from fastapi import HTTPException
from loguru import logger

from ..core import InferenceWorker
from ..models.mlx_rerank import MLX_Rerank
from ..schemas.openai import RerankRequest
from ..utils.errors import create_error_response


class MLXRerankHandler:
    """
    Handler class for making requests to the underlying MLX reranker model service.
    Provides request queuing, metrics tracking, and robust error handling with memory management.
    """

    handler_type: str = "rerank"

    def __init__(self, model_path: str, batch_size: int = 8):
        """
        Initialize the handler with the specified model path.

        Args:
            model_path (str): Path to the reranker model to load.
            batch_size (int): Number of (query, document) pairs scored per forward pass.
        """
        self.model_path = model_path
        self.batch_size = batch_size
        self.model = MLX_Rerank(model_path)
        self.model_created = int(time.time())  # Store creation time when model is loaded

        # Dedicated inference thread — keeps the event loop free during
        # blocking MLX model computation.
        self.inference_worker = InferenceWorker()

        logger.info(f"Initialized MLXRerankHandler with model path: {model_path}")

    async def get_models(self) -> list[dict[str, Any]]:
        """
        Get list of available models with their metadata.
        """
        try:
            return [
                {
                    "id": self.model_path,
                    "object": "model",
                    "created": self.model_created,
                    "owned_by": "local",
                }
            ]
        except Exception as e:
            logger.error(f"Error getting models: {e!s}")
            return []

    async def initialize(self, config: dict[str, Any]) -> None:
        """Initialize the handler and start the inference worker.

        Parameters
        ----------
        config : dict[str, Any]
            Dictionary with ``queue_size`` and ``timeout`` keys used
            to configure the inference worker's internal queue.
        """
        self.inference_worker = InferenceWorker(
            queue_size=config.get("queue_size", 100),
            timeout=config.get("timeout", 300),
        )
        self.inference_worker.start()
        logger.info("Initialized MLXRerankHandler and started inference worker")

    async def generate_rerank_response(self, request: RerankRequest) -> list[float]:
        """
        Score each document against the query.

        Args:
            request: RerankRequest with the query and candidate documents.

        Returns:
            list[float]: One relevance score per document, in input order.
        """
        try:
            return await self.inference_worker.submit(
                self.model,
                query=request.query,
                documents=list(request.documents),
                instruction=request.instruction,
                batch_size=self.batch_size,
            )

        except Exception as e:
            logger.error(f"Error in rerank generation: {e!s}")
            content = create_error_response(
                f"Failed to rerank documents: {e!s}",
                "server_error",
                HTTPStatus.INTERNAL_SERVER_ERROR,
            )
            raise HTTPException(status_code=500, detail=content)

    async def get_queue_stats(self) -> dict[str, Any]:
        """Get statistics from the inference worker.

        Returns
        -------
        dict[str, Any]
            Dictionary with ``queue_stats`` sub-dictionary.
        """
        return {
            "queue_stats": self.inference_worker.get_stats(),
        }

    async def cleanup(self) -> None:
        """Cleanup resources and stop the inference worker before shutdown.

        This method ensures all pending requests are properly completed
        and resources are released.
        """
        try:
            logger.info("Cleaning up MLXRerankHandler resources")
            if hasattr(self, "inference_worker"):
                self.inference_worker.stop()
            if hasattr(self, "model"):
                self.model.cleanup()
            # Force garbage collection
            gc.collect()
            logger.info("MLXRerankHandler cleanup completed successfully")
        except Exception as e:
            logger.error(f"Error during MLXRerankHandler cleanup: {e!s}")
            raise

    def __del__(self):
        """
        Destructor to ensure cleanup on object deletion.
        Note: Async cleanup cannot be reliably performed in __del__.
        Please use 'await cleanup()' explicitly.
        """
        if hasattr(self, "_cleaned") and self._cleaned:
            return
        # Set flag to prevent multiple cleanup attempts
        self._cleaned = True
