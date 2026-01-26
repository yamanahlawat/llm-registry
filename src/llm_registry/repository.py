"""
Repository for storing and retrieving model capabilities.
"""
from pathlib import Path

from .models import ModelCapabilities, Provider
from .utils import (
    get_user_data_dir as _get_user_data_dir,
    get_user_models_file as _get_user_models_file,
    load_user_models,
    save_user_models,
    normalize_provider_value,
)

# re-export for tests
get_user_data_dir = _get_user_data_dir

# global override hook used ONLY by repository
_ACTIVE_DATA_DIR: Path | None = None


def get_user_models_file() -> Path:
    """
    Repository-aware override.
    """
    if _ACTIVE_DATA_DIR is None:
        return _get_user_models_file()
    return _ACTIVE_DATA_DIR / "models.json"


class CapabilityRepository:
    """
    Repository for storing and retrieving model capabilities.
    """

    def __init__(self, data_dir: Path | None = None):
        global _ACTIVE_DATA_DIR

        self.data_dir = data_dir if data_dir is not None else get_user_data_dir()
        self.data_dir.mkdir(parents=True, exist_ok=True)

        _ACTIVE_DATA_DIR = self.data_dir

        # ensure file exists + reset cache
        get_user_models_file()
        load_user_models.cache_clear()

        self._user_models = load_user_models()

    def save_model_capabilities(self, capabilities: ModelCapabilities) -> Path:
        model_id = capabilities.model_id
        model_data = capabilities.model_dump(exclude={"model_id"})

        self._user_models["models"][model_id] = model_data
        save_user_models(self._user_models)

        return get_user_models_file()

    def get_model_capabilities(self, provider: Provider, model_id: str) -> ModelCapabilities | None:
        if model_id not in self._user_models["models"]:
            return None

        model_data = self._user_models["models"][model_id]
        provider_value = normalize_provider_value(provider)

        if provider_value not in model_data["providers"]:
            return None

        return ModelCapabilities.model_validate(
            {**model_data, "model_id": model_id}
        )

    def list_models(self, provider: Provider | None = None) -> list[ModelCapabilities]:
        result = []

        for model_id, model_data in self._user_models["models"].items():
            if provider:
                provider_value = normalize_provider_value(provider)
                if provider_value not in model_data["providers"]:
                    continue

            result.append(
                ModelCapabilities.model_validate(
                    {**model_data, "model_id": model_id}
                )
            )

        return result

    def delete_model(self, provider: Provider, model_id: str) -> bool:
        if model_id not in self._user_models["models"]:
            return False

        model_data = self._user_models["models"][model_id]
        provider_value = normalize_provider_value(provider)

        if provider_value not in model_data["providers"]:
            return False

        if len(model_data["providers"]) == 1:
            del self._user_models["models"][model_id]
        else:
            model_data["providers"].remove(provider_value)

        save_user_models(self._user_models)
        return True
