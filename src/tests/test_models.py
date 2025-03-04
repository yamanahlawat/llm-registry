"""
Tests for model validation and behavior.
"""

import pytest
from pydantic import ValidationError

from llm_registry.models import ApiParams, Features, ModelCapabilities, Provider, TokenCost


def test_provider_enum():
    """Test Provider enum values."""
    assert Provider.OPENAI.value == "openai"
    assert Provider.ANTHROPIC.value == "anthropic"
    assert Provider.GOOGLE.value == "google"
    assert Provider.OTHER.value == "other"


def test_token_cost_validation_valid():
    """Test that valid TokenCost objects can be created."""
    # Basic token cost
    cost = TokenCost(input_cost=1.0, output_cost=2.0)
    assert cost.input_cost == 1.0
    assert cost.output_cost == 2.0
    assert cost.cache_input_cost is None
    assert cost.cache_output_cost is None

    # With cache costs
    cost = TokenCost(
        input_cost=1.0,
        output_cost=2.0,
        cache_input_cost=0.5,
        cache_output_cost=1.0,
        context_window=8000,
        training_cutoff="2023-12",
    )
    assert cost.cache_input_cost == 0.5
    assert cost.cache_output_cost == 1.0
    assert cost.context_window == 8000
    assert cost.training_cutoff == "2023-12"


def test_token_cost_validation_invalid_cache_costs():
    """Test that TokenCost validation fails when cache costs exceed normal costs."""
    # Cache input cost higher than input cost
    with pytest.raises(ValidationError):
        TokenCost(input_cost=1.0, output_cost=2.0, cache_input_cost=1.5)

    # Cache output cost higher than output cost
    with pytest.raises(ValidationError):
        TokenCost(input_cost=1.0, output_cost=2.0, cache_output_cost=2.5)


def test_api_params_defaults():
    """Test that ApiParams uses correct defaults."""
    params = ApiParams()
    assert params.max_tokens is False
    assert params.temperature is False
    assert params.top_p is False
    assert params.stream is False


def test_features_defaults():
    """Test that Features uses correct defaults."""
    features = Features()
    assert features.vision is False
    assert features.tools is False
    assert features.json_mode is False
    assert features.system_prompt is False


def test_model_capabilities_validation():
    """Test that ModelCapabilities validates correctly."""
    # Minimal valid model
    model = ModelCapabilities(
        model_id="test-model",
        providers=[Provider.OPENAI],
    )
    assert model.model_id == "test-model"
    assert model.providers == [Provider.OPENAI]
    assert model.model_family is None
    assert model.base_model is None
    assert model.api_params.stream is False
    assert model.features.tools is False
    assert model.token_costs is None

    # Full model with all fields
    model = ModelCapabilities(
        model_id="test-model-full",
        providers=[Provider.OPENAI, Provider.AZURE],
        model_family="test-family",
        base_model="base-model",
        api_params=ApiParams(max_tokens=True, temperature=True, top_p=True, stream=True),
        features=Features(vision=True, tools=True, json_mode=True, system_prompt=True),
        token_costs=TokenCost(
            input_cost=1.0,
            output_cost=2.0,
            cache_input_cost=0.5,
            cache_output_cost=1.0,
            context_window=8000,
            training_cutoff="2023-12",
        ),
    )
    assert model.model_id == "test-model-full"
    assert len(model.providers) == 2
    assert Provider.OPENAI in model.providers
    assert Provider.AZURE in model.providers
    assert model.model_family == "test-family"
    assert model.base_model == "base-model"
    assert model.api_params.max_tokens is True
    assert model.api_params.temperature is True
    assert model.api_params.top_p is True
    assert model.api_params.stream is True
    assert model.features.vision is True
    assert model.features.tools is True
    assert model.features.json_mode is True
    assert model.features.system_prompt is True
    assert model.token_costs.input_cost == 1.0
    assert model.token_costs.output_cost == 2.0
    assert model.token_costs.cache_input_cost == 0.5
    assert model.token_costs.cache_output_cost == 1.0
    assert model.token_costs.context_window == 8000
    assert model.token_costs.training_cutoff == "2023-12"


def test_model_capabilities_str_representation():
    """Test string representation of ModelCapabilities."""
    model = ModelCapabilities(
        model_id="test-model",
        providers=[Provider.OPENAI, Provider.AZURE],
        model_family="test-family",
        features=Features(vision=True, tools=False, json_mode=True, system_prompt=False),
    )
    str_repr = str(model)
    assert "Providers: openai, azure" in str_repr
    assert "Family: test-family" in str_repr
    assert "Vision=True" in str_repr
    assert "Tools=False" in str_repr
    assert "JSON=True" in str_repr
    assert "System=False" in str_repr


def test_model_capabilities_from_dict():
    """Test creating ModelCapabilities from a dictionary."""
    model_dict = {
        "model_id": "from-dict",
        "providers": ["openai", "azure"],
        "model_family": "test-family",
        "api_params": {"max_tokens": True, "stream": True},
        "features": {"vision": True, "tools": True},
        "token_costs": {"input_cost": 1.0, "output_cost": 2.0},
    }
    model = ModelCapabilities.model_validate(model_dict)
    assert model.model_id == "from-dict"
    assert Provider.OPENAI in model.providers
    assert Provider.AZURE in model.providers
    assert model.model_family == "test-family"
    assert model.api_params.max_tokens is True
    assert model.api_params.stream is True
    assert model.features.vision is True
    assert model.features.tools is True
    assert model.token_costs.input_cost == 1.0
    assert model.token_costs.output_cost == 2.0
