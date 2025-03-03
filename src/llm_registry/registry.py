"""
Registry for managing model capabilities data from both remote and local sources.
"""

import json
import time
from pathlib import Path
from typing import Dict, List

import httpx

from .models import ModelCapabilities, Provider

# Default remote URL
REMOTE_JSON_URL = "https://gist.githubusercontent.com/yamanahlawat/707a6ea05b079669347745a4484f5c6d/raw/models.json"


class CapabilityRegistry:
    """
    Registry for managing model capabilities data.
    """

    def __init__(
        self,
        remote_url: str = REMOTE_JSON_URL,
        cache_ttl: int = 3600,  # 1 hour
        offline_mode: bool = False,
        timeout: float = 10.0,  # 10 second timeout
    ):
        """
        Initialize the registry.
        Args:
            remote_url: URL to fetch remote JSON data from
            cache_ttl: Time in seconds to cache remote data
            offline_mode: If True, only use packaged data
            timeout: Timeout in seconds for HTTP requests
        """
        self.remote_url = remote_url
        self.cache_ttl = cache_ttl
        self.offline_mode = offline_mode
        self.timeout = timeout
        self._cache: Dict[str, ModelCapabilities] | None = None
        self._last_fetch: float | None = None
        self._client = httpx.Client(timeout=timeout)

    def get_models(self, provider: Provider | None = None) -> List[ModelCapabilities]:
        """
        Get all model capabilities, optionally filtered by provider.
        Args:
            provider: Optional provider to filter by
        Returns:
            List of ModelCapabilities objects
        """
        data = self._get_data()
        models = []

        for model_id, model_data in data["models"].items():
            if provider and provider not in model_data["providers"]:
                continue
            # Add model_id to data for validation
            model_cap = ModelCapabilities.model_validate({**model_data, "model_id": model_id})
            models.append(model_cap)

        return models

    def get_model(self, provider: Provider, model_id: str) -> ModelCapabilities | None:
        """
        Get capabilities for a specific model.
        Args:
            provider: The model provider
            model_id: The model identifier
        Returns:
            ModelCapabilities if found, None otherwise
        """
        data = self._get_data()

        if model_id not in data["models"]:
            return None

        model_data = data["models"][model_id]
        if provider.value not in model_data["providers"]:
            return None

        # Add model_id to data for validation
        return ModelCapabilities.model_validate({**model_data, "model_id": model_id})

    def _get_data(self) -> dict:
        """
        Get model data, trying remote first then falling back to packaged.
        """
        if not self.offline_mode:
            try:
                return self._get_remote_data()
            except (httpx.HTTPError, Exception):
                # Any failure, fall back to packaged data
                pass

        return self._get_packaged_data()

    def _get_remote_data(self) -> dict:
        """Get data from remote JSON."""
        now = time.time()

        # Use cache if valid
        if self._cache and self._last_fetch and (now - self._last_fetch < self.cache_ttl):
            return self._cache

        # Fetch fresh data
        response = self._client.get(self.remote_url)
        response.raise_for_status()

        self._cache = response.json()
        self._last_fetch = now
        return self._cache

    def _get_packaged_data(self) -> dict:
        """
        Get data from package JSON.
        """
        path = Path(__file__).parent / "data" / "models.json"
        with open(path) as f:
            return json.load(f)

    def __del__(self):
        """
        Ensure client is closed on cleanup.
        """
        if hasattr(self, "_client"):
            self._client.close()
