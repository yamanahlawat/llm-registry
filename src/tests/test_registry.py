"""
Tests for the CapabilityRegistry class.
"""

from unittest.mock import patch

import pytest

from llm_registry import CapabilityRegistry
from llm_registry.exceptions import ModelNotFoundError
from llm_registry.models import Provider


@pytest.fixture
def mock_package_models():
    return {
        "models": {
            "gpt-4o": {
                "providers": ["openai"],
                "model_family": "gpt-4",
                "api_params": {"stream": True},
                "features": {"vision": True, "tools": True, "json_mode": True, "system_prompt": True},
                "token_costs": {
                    "input_cost": 2.5,
                    "output_cost": 10,
                    "context_window": 128000,
                    "training_cutoff": "2024-01",
                },
            },
            "claude-3-haiku-latest": {
                "providers": ["anthropic"],
                "model_family": "claude-3",
                "api_params": {"stream": True},
                "features": {"vision": True, "tools": True, "json_mode": True, "system_prompt": True},
                "token_costs": {
                    "input_cost": 15,
                    "output_cost": 75,
                    "context_window": 200000,
                    "training_cutoff": "2024-02",
                },
            },
        }
    }


@pytest.fixture
def mock_user_models():
    return {
        "models": {
            "gpt-4o": {
                "providers": ["openai", "azure"],  # Added Azure provider
                "model_family": "gpt-4",
                "api_params": {"stream": True},
                "features": {"vision": True, "tools": True, "json_mode": True, "system_prompt": True},
                "token_costs": {
                    "input_cost": 2.0,  # Different cost from package model
                    "output_cost": 8.0,
                    "context_window": 128000,
                    "training_cutoff": "2024-01",
                },
            },
            "custom-model": {
                "providers": ["openai"],
                "model_family": "custom",
                "api_params": {"stream": False},
                "features": {"vision": False, "tools": False, "json_mode": False, "system_prompt": False},
                "token_costs": {
                    "input_cost": 1.0,
                    "output_cost": 2.0,
                    "context_window": 8000,
                    "training_cutoff": "2023-12",
                },
            },
        }
    }


@pytest.fixture
def registry(mock_package_models, mock_user_models):
    with patch("llm_registry.registry.load_package_models", return_value=mock_package_models):
        with patch("llm_registry.registry.load_user_models", return_value=mock_user_models):
            yield CapabilityRegistry()


def test_get_model_exists_in_user_models(registry):
    """Test retrieving a model that exists in user models."""
    model = registry.get_model("gpt-4o")
    assert model.model_id == "gpt-4o"
    assert model.token_costs.input_cost == 2.0  # User model value
    assert Provider.AZURE in model.providers  # User model has AZURE provider


def test_get_model_exists_in_package_models(registry):
    """Test retrieving a model that exists only in package models."""
    model = registry.get_model("claude-3-haiku-latest")
    assert model.model_id == "claude-3-haiku-latest"
    assert model.token_costs.input_cost == 15.0
    assert Provider.ANTHROPIC in model.providers


def test_get_model_not_found(registry):
    """Test retrieving a non-existent model."""
    with pytest.raises(ModelNotFoundError) as exc_info:
        registry.get_model("nonexistent-model")
    assert str(exc_info.value) == "Model 'nonexistent-model' not found in registry"


def test_get_models_no_filter(registry):
    """Test retrieving all models with no provider filter."""
    models = registry.get_models()
    assert len(models) == 3
    model_ids = [model.model_id for model in models]
    assert "gpt-4o" in model_ids
    assert "claude-3-haiku-latest" in model_ids
    assert "custom-model" in model_ids


def test_get_models_with_provider_filter(registry):
    """Test retrieving models filtered by provider."""
    models = registry.get_models(provider=Provider.OPENAI)
    assert len(models) == 2
    model_ids = [model.model_id for model in models]
    assert "gpt-4o" in model_ids
    assert "custom-model" in model_ids
    assert "claude-3-haiku-latest" not in model_ids

    models = registry.get_models(provider=Provider.ANTHROPIC)
    assert len(models) == 1
    assert models[0].model_id == "claude-3-haiku-latest"

    models = registry.get_models(provider=Provider.AZURE)
    assert len(models) == 1
    assert models[0].model_id == "gpt-4o"


def test_user_models_override_package_models(registry):
    """Test that user models override package models with the same ID."""
    model = registry.get_model("gpt-4o")
    assert model.token_costs.input_cost == 2.0  # User model value, not 2.5 from package
    assert len(model.providers) == 2  # User model has both providers
    assert Provider.OPENAI in model.providers
    assert Provider.AZURE in model.providers


def test_get_model_base_model_fallback(registry):
    """Test fallback to base model when model_id contains a colon."""
    # Add a base model to user models
    base_model_id = "custom-model"
    variant_model_id = f"{base_model_id}:variant"
    # Should return the base model if variant does not exist
    model = registry.get_model(variant_model_id)
    assert model.model_id == base_model_id
    assert model.token_costs.input_cost == 1.0


def test_get_model_base_model_fallback_not_found(registry):
    """Test that ModelNotFoundError is raised if neither the full nor base model exists."""
    with pytest.raises(ModelNotFoundError):
        registry.get_model("nonexistent:variant")


def test_get_model_not_found_user_and_package_and_base(registry, mock_package_models, mock_user_models):
    """Test ModelNotFoundError is raised when model is missing from user, package, and base fallback."""
    # Remove all models to ensure no match
    mock_package_models["models"].clear()
    mock_user_models["models"].clear()
    with (
        patch("llm_registry.registry.load_package_models", return_value=mock_package_models),
        patch("llm_registry.registry.load_user_models", return_value=mock_user_models),
    ):
        reg = CapabilityRegistry()
        with pytest.raises(ModelNotFoundError) as exc_info:
            reg.get_model("totally-missing:variant")
        assert "totally-missing:variant" in str(exc_info.value)


def test_get_models_provider_no_match(registry):
    """Test that filtering by a provider not present in any model returns an empty list."""

    class FakeProvider(str):
        value = "notarealprovider"

    models = registry.get_models(provider=FakeProvider)
    assert models == []


def test_model_with_empty_providers(registry, mock_user_models):
    """Test that a model with an empty providers list is not returned for any provider filter."""
    mock_user_models["models"]["empty-provider-model"] = {
        "providers": [],
        "model_family": "empty",
        "api_params": {"stream": True},
        "features": {"vision": False, "tools": False, "json_mode": False, "system_prompt": False},
        "token_costs": {
            "input_cost": 0.0,
            "output_cost": 0.0,
            "context_window": 1000,
            "training_cutoff": "2025-01",
        },
    }
    with patch("llm_registry.registry.load_user_models", return_value=mock_user_models):
        reg = CapabilityRegistry()
        models = reg.get_models(provider=Provider.OPENAI)
        assert all(m.model_id != "empty-provider-model" for m in models)


def test_user_model_overrides_package_providers(registry, mock_package_models, mock_user_models):
    """Test that user model providers override package model providers for the same ID."""
    # Add a package model with a different provider
    mock_package_models["models"]["override-model"] = {
        "providers": ["anthropic"],
        "model_family": "override",
        "api_params": {"stream": True},
        "features": {"vision": True, "tools": True, "json_mode": True, "system_prompt": True},
        "token_costs": {
            "input_cost": 1.0,
            "output_cost": 2.0,
            "context_window": 1000,
            "training_cutoff": "2025-01",
        },
    }
    # Add a user model with the same ID but different providers
    mock_user_models["models"]["override-model"] = {
        "providers": ["openai"],
        "model_family": "override",
        "api_params": {"stream": True},
        "features": {"vision": True, "tools": True, "json_mode": True, "system_prompt": True},
        "token_costs": {
            "input_cost": 1.0,
            "output_cost": 2.0,
            "context_window": 1000,
            "training_cutoff": "2025-01",
        },
    }
    with (
        patch("llm_registry.registry.load_package_models", return_value=mock_package_models),
        patch("llm_registry.registry.load_user_models", return_value=mock_user_models),
    ):
        reg = CapabilityRegistry()
        model = reg.get_model("override-model")
        assert Provider.OPENAI in model.providers
        assert Provider.ANTHROPIC not in model.providers


def test_model_with_missing_context_window_and_training_cutoff(registry, mock_user_models):
    """Test that models with missing context_window or training_cutoff are still loaded."""
    mock_user_models["models"]["partial-fields-model"] = {
        "providers": ["openai"],
        "model_family": "partial",
        "api_params": {"stream": True},
        "features": {"vision": True, "tools": True, "json_mode": True, "system_prompt": True},
        "token_costs": {
            "input_cost": 1.0,
            "output_cost": 2.0,
            # context_window and training_cutoff omitted
        },
    }
    with patch("llm_registry.registry.load_user_models", return_value=mock_user_models):
        reg = CapabilityRegistry()
        model = next(m for m in reg.get_models() if m.model_id == "partial-fields-model")
        assert model.token_costs is not None
        assert getattr(model.token_costs, "input_cost", None) == 1.0
        assert getattr(model.token_costs, "output_cost", None) == 2.0
        assert getattr(model.token_costs, "context_window", None) is None
        assert getattr(model.token_costs, "training_cutoff", None) is None
