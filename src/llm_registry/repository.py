"""
Repository for storing and retrieving model capabilities.
"""

import json
import os
from pathlib import Path

from .models import ModelCapabilities, Provider

# Default paths
DEFAULT_DATA_DIR = Path.home() / ".llm-registry"
DEFAULT_MODELS_DIR = DEFAULT_DATA_DIR / "models"


class CapabilityRepository:
    """
    Repository for storing and retrieving model capabilities.
    """

    def __init__(self, data_dir: Path | None = None):
        """
        Initialize the repository with a data directory.
        Args:
            data_dir: Directory to store model capability data.
                     If None, defaults to ~/.llm-registry
        """
        if data_dir is None:
            self.data_dir = DEFAULT_MODELS_DIR
        else:
            self.data_dir = data_dir / "models"

        self.data_dir.mkdir(parents=True, exist_ok=True)

        # Cache for loaded capabilities
        self._capabilities_cache: dict[str, ModelCapabilities] = {}

    def _get_model_path(self, provider: Provider, model_id: str) -> Path:
        """
        Get the file path for a model's capability data.
        Args:
            provider: The model provider
            model_id: The model identifier
        Returns:
            Path to the model's JSON file
        """
        provider_value = provider.value if isinstance(provider, Provider) else provider
        return self.data_dir / f"{provider_value}_{model_id}.json"

    def save_model_capabilities(self, capabilities: ModelCapabilities) -> Path:
        """
        Save model capabilities to the repository.
        Args:
            capabilities: The capabilities to save
        Returns:
            Path where the capabilities were saved
        """
        # Save a copy for each provider
        paths = []
        for provider in capabilities.providers:
            file_path = self._get_model_path(provider, capabilities.model_id)
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(capabilities.model_dump_json(indent=2))

            # Update cache
            cache_key = f"{provider.value}_{capabilities.model_id}"
            self._capabilities_cache[cache_key] = capabilities
            paths.append(file_path)

        return paths[0]  # Return the first path for backward compatibility

    def get_model_capabilities(self, provider: Provider, model_id: str) -> ModelCapabilities | None:
        """
        Get capabilities for a specific model.
        Args:
            provider: The model provider
            model_id: The model identifier
        Returns:
            ModelCapabilities if found, None otherwise
        """
        provider_value = provider.value if isinstance(provider, Provider) else provider
        cache_key = f"{provider_value}_{model_id}"

        # Check cache first
        if cache_key in self._capabilities_cache:
            cap = self._capabilities_cache[cache_key]
            if provider in cap.providers:
                return cap

        # Check file system
        file_path = self._get_model_path(provider, model_id)
        if not file_path.exists():
            return None

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                capabilities = ModelCapabilities.model_validate(data)

                # Verify provider is in the model's providers
                if provider not in capabilities.providers:
                    return None

                # Update cache
                self._capabilities_cache[cache_key] = capabilities
                return capabilities
        except (json.JSONDecodeError, Exception) as e:
            print(f"Error loading model capabilities: {e}")
            return None

    def list_models(self, provider: Provider | None = None) -> list[ModelCapabilities]:
        """
        List all models in the repository, optionally filtering by provider.
        Args:
            provider: Optional provider to filter by
        Returns:
            List of ModelCapabilities objects
        """
        result = []
        seen_models = set()  # Track unique model IDs

        # Filter pattern for filenames
        pattern = f"{provider.value}_*.json" if provider else "*.json"

        for file_path in self.data_dir.glob(pattern):
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    capabilities = ModelCapabilities.model_validate(data)

                    # Skip if we've already seen this model
                    if capabilities.model_id in seen_models:
                        continue

                    # If provider specified, check if it's in the model's providers
                    if provider and provider not in capabilities.providers:
                        continue

                    result.append(capabilities)
                    seen_models.add(capabilities.model_id)

                    # Update cache for each provider
                    for p in capabilities.providers:
                        cache_key = f"{p.value}_{capabilities.model_id}"
                        self._capabilities_cache[cache_key] = capabilities
            except Exception as e:
                print(f"Error loading {file_path}: {e}")

        return result

    def delete_model(self, provider: Provider, model_id: str) -> bool:
        """
        Delete a model from the repository.
        Args:
            provider: The model provider
            model_id: The model identifier
        Returns:
            True if deleted, False if not found
        """
        file_path = self._get_model_path(provider, model_id)
        if not file_path.exists():
            return False

        try:
            os.remove(file_path)

            # Remove from cache
            provider_value = provider.value if isinstance(provider, Provider) else provider
            cache_key = f"{provider_value}_{model_id}"
            if cache_key in self._capabilities_cache:
                del self._capabilities_cache[cache_key]

            return True
        except Exception as e:
            print(f"Error deleting model: {e}")
            return False
