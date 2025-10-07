"""
Utility functions for the LLM registry package.
"""

import json
from functools import lru_cache
from pathlib import Path

from llm_registry.models import ApiParams, Features, ModelCapabilities, Provider, TokenCost


def create_model_capability(
    model_id: str,
    provider: Provider | str | list[Provider] | list[str],
    model_family: str | None = None,
    supports_streaming: bool = False,
    supports_tools: bool = False,
    supports_vision: bool = False,
    supports_json_mode: bool = False,
    supports_system_prompt: bool = False,
    input_cost: float | None = None,
    output_cost: float | None = None,
    context_window: int | None = None,
    training_cutoff: str | None = None,
) -> ModelCapabilities:
    """
    Helper function to create a ModelCapabilities object with less verbose syntax.
    Args:
        model_id: Model identifier
        provider: Model provider(s) - single Provider/str or list of Provider/str
        model_family: Optional model family
        supports_streaming: Whether the model supports streaming
        supports_tools: Whether the model supports tools/function calling
        supports_vision: Whether the model supports vision inputs
        supports_json_mode: Whether the model supports JSON mode
        supports_system_prompt: Whether the model supports system prompts
        input_cost: Cost per 1M input tokens
        output_cost: Cost per 1M output tokens
        context_window: Context window size in tokens
        training_cutoff: Training data cutoff date
    Returns:
        ModelCapabilities object
    """
    # Convert provider(s) to list of Provider enums
    providers = []
    provider_list = [provider] if not isinstance(provider, list) else provider

    for p in provider_list:
        if isinstance(p, str):
            try:
                providers.append(Provider(p))
            except ValueError:
                providers.append(Provider.OTHER)
        else:
            providers.append(p)

    # Create token costs if applicable
    token_costs = None
    if input_cost is not None and output_cost is not None:
        token_costs = TokenCost(
            input_cost=input_cost,
            output_cost=output_cost,
            context_window=context_window,
            training_cutoff=training_cutoff,
        )

    # Create API params and features
    api_params = ApiParams(stream=supports_streaming)
    features = Features(
        vision=supports_vision,
        tools=supports_tools,
        json_mode=supports_json_mode,
        system_prompt=supports_system_prompt,
    )

    return ModelCapabilities(
        model_id=model_id,
        providers=providers,
        model_family=model_family,
        api_params=api_params,
        features=features,
        token_costs=token_costs,
    )


def get_user_data_dir() -> Path:
    """
    Get the path to user's data directory.
    """
    data_dir = Path.home() / ".llm-registry"
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir


def get_user_models_file() -> Path:
    """
    Get the path to user's models.json file.
    """
    data_dir = get_user_data_dir()
    models_file = data_dir / "models.json"
    if not models_file.exists():
        # Create empty models file with simplified structure
        with open(models_file, "w", encoding="utf-8") as models_file_obj:
            json.dump({"models": {}}, models_file_obj, indent=2)
    return models_file


def load_package_models() -> dict:
    """
    Load models from package JSON file with caching.
    """
    packages_models_file = Path(__file__).parent / "data" / "models.json"
    with open(packages_models_file, encoding="utf-8") as f:
        return json.load(f)


@lru_cache()
def load_user_models() -> dict:
    """
    Load models from user's JSON file with caching.
    """
    models_file = get_user_models_file()
    try:
        with open(models_file, encoding="utf-8") as models_file_obj:
            return json.load(models_file_obj)
    except json.JSONDecodeError:
        # If the file is corrupted, return an empty models structure
        return {"models": {}}


def save_user_models(data: dict) -> None:
    """
    Save models to user's JSON file.
    """
    # Ensure data has the correct structure
    if "models" not in data:
        data["models"] = {}

    # Save to file
    models_file = get_user_models_file()
    with open(models_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    # Clear cache so next load reads fresh data
    load_user_models.cache_clear()


def normalize_provider_value(provider: Provider | str) -> str:
    """
    Convert Provider enum to string value if needed.

    Args:
        provider: Provider enum or string value

    Returns:
        String value of the provider
    """
    return provider.value if isinstance(provider, Provider) else provider


def is_package_model(model_id: str) -> bool:
    """
    Check if a model exists in package data.
    """
    package_data = load_package_models()
    return model_id in package_data["models"]
