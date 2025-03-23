"""
Registry for LLM model capabilities.
"""

from llm_registry.exceptions import ModelNotFoundError
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
        Get model capabilities by model ID with fallback to model family variants.
        Args:
            model_id: Model identifier (exact ID or base family name)
        Returns:
            Model capabilities
        Raises:
            ModelNotFoundError: If model not found
        """
        # Try exact match first (most efficient)
        if model_id in self._user_models["models"]:
            model_data = self._user_models["models"][model_id]
            return ModelCapabilities.model_validate({**model_data, "model_id": model_id})

        # Then check package models
        if model_id in self._package_models["models"]:
            model_data = self._package_models["models"][model_id]
            return ModelCapabilities.model_validate({**model_data, "model_id": model_id})

        # If not found AND model_id DOES contain a colon, try looking for the base model
        if ":" in model_id:
            # Extract the base model name (part before the colon)
            base_model = model_id.split(":")[0]

            # Check if the base model exists
            if base_model in self._user_models["models"]:
                model_data = self._user_models["models"][base_model]
                return ModelCapabilities.model_validate({**model_data, "model_id": base_model})

            if base_model in self._package_models["models"]:
                model_data = self._package_models["models"][base_model]
                return ModelCapabilities.model_validate({**model_data, "model_id": base_model})

        # No match found
        raise ModelNotFoundError(f"Model '{model_id}' not found in registry")

    def get_models(self, provider: Provider | None = None) -> list[ModelCapabilities]:
        """
        Get all model capabilities, optionally filtered by provider.
        Args:
            provider: Optional provider to filter by.
        Returns:
            List of model capabilities.
        """
        # Combine package and user models
        all_models: dict[str, dict] = {**self._package_models["models"]}
        all_models |= self._user_models["models"]

        return [
            ModelCapabilities.model_validate({**model_data, "model_id": model_id})
            for model_id, model_data in all_models.items()
            if not provider or provider.value in model_data["providers"]
        ]
