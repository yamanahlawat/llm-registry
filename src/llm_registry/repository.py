"""
Repository for storing and retrieving model capabilities.
"""

from pathlib import Path

from .models import ModelCapabilities, Provider
from .utils import get_user_data_dir, get_user_models_file, load_user_models, normalize_provider_value, save_user_models


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
        self.data_dir = get_user_data_dir() if data_dir is None else data_dir
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # Load user models
        self._user_models = load_user_models()

    def save_model_capabilities(self, capabilities: ModelCapabilities) -> Path:
        """
        Save model capabilities to the repository.
        Args:
            capabilities: The capabilities to save
        Returns:
            Path where the capabilities were saved
        """
        # Update the user models dictionary
        model_id = capabilities.model_id
        model_data = capabilities.model_dump(exclude={"model_id"})

        # Add or update the model in the user models dictionary
        self._user_models["models"][model_id] = model_data

        # Save to file
        save_user_models(self._user_models)

        # Return the path to the models file
        return get_user_models_file()

    def get_model_capabilities(self, provider: Provider, model_id: str) -> ModelCapabilities | None:
        """
        Get capabilities for a specific model.
        Args:
            provider: The model provider
            model_id: The model identifier
        Returns:
            ModelCapabilities if found, None otherwise
        """
        # Check if the model exists
        if model_id not in self._user_models["models"]:
            return None

        # Get the model data
        model_data = self._user_models["models"][model_id]

        # Verify provider is in the model's providers
        provider_value = normalize_provider_value(provider)
        if provider_value not in model_data["providers"]:
            return None

        # Create and return the ModelCapabilities object
        return ModelCapabilities.model_validate({**model_data, "model_id": model_id})

    def list_models(self, provider: Provider | None = None) -> list[ModelCapabilities]:
        """
        List all models in the repository, optionally filtering by provider.
        Args:
            provider: Optional provider to filter by
        Returns:
            List of ModelCapabilities objects
        """
        result = []

        # Filter models by provider if specified
        for model_id, model_data in self._user_models["models"].items():
            # If provider is specified, check if it's in the model's providers
            if provider:
                provider_value = normalize_provider_value(provider)
                if provider_value not in model_data["providers"]:
                    continue

            # Create and add the ModelCapabilities object
            capabilities = ModelCapabilities.model_validate({**model_data, "model_id": model_id})
            result.append(capabilities)

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
        # Check if the model exists
        if model_id not in self._user_models["models"]:
            return False

        # Get the model data
        model_data = self._user_models["models"][model_id]

        # Verify provider is in the model's providers
        provider_value = normalize_provider_value(provider)
        if provider_value not in model_data["providers"]:
            return False

        # If there's only one provider, delete the model
        if len(model_data["providers"]) == 1:
            del self._user_models["models"][model_id]
        else:
            # Otherwise, remove the provider from the list
            model_data["providers"].remove(provider_value)

        # Save the updated models
        save_user_models(self._user_models)

        return True
