"""Tests for the CapabilityRegistry class."""

import json
from unittest.mock import patch

import pytest

from llm_registry import CapabilityRegistry, Provider


@pytest.fixture
def sample_package_models(tmp_path):
    """Create a sample package models.json file."""
    models_data = {
        "version": "1.0.0",
        "last_updated": "2024-03-02",
        "models": {
            "o3-mini": {
                "providers": ["openai"],
                "model_family": "o3",
                "api_params": {
                    "max_tokens": True,
                    "temperature": True,
                    "top_p": True,
                    "frequency_penalty": True,
                    "presence_penalty": True,
                    "stream": True,
                },
                "features": {"tools": True, "vision": False, "json_mode": True, "system_prompt": True},
                "token_costs": {
                    "input_cost": 0.2,
                    "output_cost": 0.4,
                    "context_window": 200000,
                    "training_cutoff": "2024-01",
                },
            }
        },
    }

    json_file = tmp_path / "models.json"
    with open(json_file, "w") as f:
        json.dump(models_data, f)
    return json_file


@pytest.fixture
def sample_user_models(tmp_path):
    """Create a sample user models.json file."""
    models_data = {
        "models": {
            "custom-model": {
                "providers": ["anthropic"],
                "model_family": "custom",
                "api_params": {"max_tokens": True, "temperature": True, "stream": True},
                "features": {"tools": False, "vision": True, "json_mode": False, "system_prompt": True},
                "token_costs": {
                    "input_cost": 0.1,
                    "output_cost": 0.2,
                    "cache_input_cost": 0.05,
                    "cache_output_cost": 0.1,
                    "context_window": 100000,
                    "training_cutoff": "2024-02",
                },
            }
        }
    }

    json_file = tmp_path / "models.json"
    with open(json_file, "w") as f:
        json.dump(models_data, f)
    return json_file


def test_get_models(sample_package_models, sample_user_models):
    """Test getting all models."""
    with (
        patch("llm_registry.registry.CapabilityRegistry._load_package_models") as mock_package,
        patch("llm_registry.registry.CapabilityRegistry._load_user_models") as mock_user,
    ):
        mock_package.return_value = json.loads(sample_package_models.read_text())
        mock_user.return_value = json.loads(sample_user_models.read_text())

        registry = CapabilityRegistry()
        models = registry.get_models()

        assert len(models) == 2
        model_ids = {m.model_id for m in models}
        assert model_ids == {"o3-mini", "custom-model"}


def test_get_models_by_provider(sample_package_models, sample_user_models):
    """Test filtering models by provider."""
    with (
        patch("llm_registry.registry.CapabilityRegistry._load_package_models") as mock_package,
        patch("llm_registry.registry.CapabilityRegistry._load_user_models") as mock_user,
    ):
        mock_package.return_value = json.loads(sample_package_models.read_text())
        mock_user.return_value = json.loads(sample_user_models.read_text())

        registry = CapabilityRegistry()

        # Test OpenAI models
        models = registry.get_models(provider=Provider.OPENAI)
        assert len(models) == 1
        assert models[0].model_id == "o3-mini"

        # Test Anthropic models
        models = registry.get_models(provider=Provider.ANTHROPIC)
        assert len(models) == 1
        assert models[0].model_id == "custom-model"

        # Test non-existent provider
        models = registry.get_models(provider=Provider.COHERE)
        assert len(models) == 0


def test_get_model(sample_package_models, sample_user_models):
    """Test getting a specific model by ID."""
    with (
        patch("llm_registry.registry.CapabilityRegistry._load_package_models") as mock_package,
        patch("llm_registry.registry.CapabilityRegistry._load_user_models") as mock_user,
    ):
        mock_package.return_value = json.loads(sample_package_models.read_text())
        mock_user.return_value = json.loads(sample_user_models.read_text())

        registry = CapabilityRegistry()

        # Test getting package model
        model = registry.get_model("o3-mini")
        assert model is not None
        assert model.model_id == "o3-mini"
        assert Provider.OPENAI in model.providers
        assert model.model_family == "o3"

        # Test getting user model
        model = registry.get_model("custom-model")
        assert model is not None
        assert model.model_id == "custom-model"
        assert Provider.ANTHROPIC in model.providers
        assert model.model_family == "custom"

        # Test non-existent model
        with pytest.raises(KeyError) as exc:
            registry.get_model("non-existent")
        assert "not found" in str(exc.value)


def test_user_model_override(sample_package_models):
    """Test that user models override package models."""
    # Create user model with same ID as package model but different data
    user_data = {
        "models": {
            "o3-mini": {
                "providers": ["anthropic"],
                "model_family": "custom",
                "api_params": {"max_tokens": True, "temperature": True, "stream": True},
                "features": {"tools": True, "vision": True, "json_mode": True, "system_prompt": True},
                "token_costs": {
                    "input_cost": 0.1,
                    "output_cost": 0.2,
                    "context_window": 100000,
                    "training_cutoff": "2024-02",
                },
            }
        }
    }

    with (
        patch("llm_registry.registry.CapabilityRegistry._load_package_models") as mock_package,
        patch("llm_registry.registry.CapabilityRegistry._load_user_models") as mock_user,
    ):
        mock_package.return_value = json.loads(sample_package_models.read_text())
        mock_user.return_value = user_data

        registry = CapabilityRegistry()
        model = registry.get_model("o3-mini")

        # Should have user model data
        assert model.providers == [Provider.ANTHROPIC]
        assert model.model_family == "custom"
        assert model.token_costs.input_cost == 0.1
