"""
Registry for LLM model capabilities.
"""

from llm_registry.models import ModelCapabilities, Provider
from llm_registry.utils import load_package_models, load_user_models


class CapabilityRegistry:
    """
    Registry for LLM model capabilities.
    """

    def __init__(self) -> None:
        """
        Initialize the registry.
        """
        self._package_models = load_package_models()
        self._user_models = load_user_models()

    def get_model(self, model_id: str) -> ModelCapabilities:
        """
        Get model capabilities by model ID.
        Args:
            model_id: Model identifier.
        Returns:
            Model capabilities.
        Raises:
            KeyError: If model not found.
        """
        # Check user models first (they take precedence)
        if model_id in self._user_models["models"]:
            model_data = self._user_models["models"][model_id]
            return ModelCapabilities.model_validate({**model_data, "model_id": model_id})

        # Then check package models
        if model_id in self._package_models["models"]:
            model_data = self._package_models["models"][model_id]
            return ModelCapabilities.model_validate({**model_data, "model_id": model_id})

        raise KeyError(f"Model '{model_id}' not found")

    def get_models(self, provider: Provider | None = None) -> list[ModelCapabilities]:
        """
        Get all model capabilities, optionally filtered by provider.
        Args:
            provider: Optional provider to filter by.
        Returns:
            List of model capabilities.
        """
        models = []

        # Combine package and user models
        all_models: dict[str, dict] = {**self._package_models["models"]}
        all_models.update(self._user_models["models"])  # User models override package models

        # Convert to ModelCapabilities objects
        for model_id, model_data in all_models.items():
            if not provider or provider.value in model_data["providers"]:
                models.append(ModelCapabilities.model_validate({**model_data, "model_id": model_id}))

        return models
