"""
Tests for the CapabilityRegistry class.
"""

from unittest.mock import patch

import pytest

from llm_registry import CapabilityRegistry
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
    with pytest.raises(KeyError):
        registry.get_model("nonexistent-model")


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
