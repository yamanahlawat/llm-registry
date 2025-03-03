"""Tests for the CapabilityRegistry class."""

import json
from pathlib import Path
from unittest.mock import Mock, patch

import httpx
import pytest
from llm_registry import CapabilityRegistry, Provider


@pytest.fixture
def sample_models_json(tmp_path):
    """Create a sample models.json file."""
    models_data = {
        "version": "1.0.0",
        "last_updated": "2024-03-02",
        "models": {
            "o3-mini": {
                "providers": ["openai"],
                "model_family": "o3",
                "api_params": {
                    "frequency_penalty": True,
                    "max_tokens": True,
                    "n": True,
                    "presence_penalty": True,
                    "temperature": True,
                    "top_p": True
                },
                "features": {
                    "streaming": True,
                    "tools": True,
                    "vision": False,
                    "json_mode": True,
                    "system_prompt": True
                },
                "token_costs": {
                    "input_cost": 0.2,
                    "output_cost": 0.4,
                    "context_window": 200000,
                    "training_cutoff": "2024-01"
                }
            }
        }
    }

    json_file = tmp_path / "models.json"
    with open(json_file, "w") as f:
        json.dump(models_data, f)
    return json_file


def test_get_models_offline(sample_models_json):
    """Test getting models in offline mode."""
    with patch("llm_registry.registry.CapabilityRegistry._get_packaged_data") as mock_get_data:
        mock_get_data.return_value = json.loads(sample_models_json.read_text())
        registry = CapabilityRegistry(offline_mode=True)
        models = registry.get_models()

        assert len(models) == 1
        model = models[0]
        assert model.model_id == "o3-mini"
        assert Provider.OPENAI in model.providers
        assert model.features.vision is False
        assert model.token_costs.input_cost == 0.2


def test_get_models_by_provider(sample_models_json):
    """Test filtering models by provider."""
    with patch("llm_registry.registry.CapabilityRegistry._get_packaged_data") as mock_get_data:
        mock_get_data.return_value = json.loads(sample_models_json.read_text())
        registry = CapabilityRegistry(offline_mode=True)

        # Should find the OpenAI model
        models = registry.get_models(provider=Provider.OPENAI)
        assert len(models) == 1

        # Should not find any Anthropic models
        models = registry.get_models(provider=Provider.ANTHROPIC)
        assert len(models) == 0


def test_get_specific_model(sample_models_json):
    """Test getting a specific model by ID."""
    with patch("llm_registry.registry.CapabilityRegistry._get_packaged_data") as mock_get_data:
        mock_get_data.return_value = json.loads(sample_models_json.read_text())
        registry = CapabilityRegistry(offline_mode=True)

        # Should find existing model
        model = registry.get_model(Provider.OPENAI, "o3-mini")
        assert model is not None
        assert model.model_id == "o3-mini"

        # Should return None for non-existent model
        model = registry.get_model(Provider.OPENAI, "non-existent")
        assert model is None


@pytest.mark.asyncio
async def test_remote_fallback(sample_models_json):
    """Test fallback to packaged data when remote fails."""
    with patch("llm_registry.registry.CapabilityRegistry._get_packaged_data") as mock_get_data:
        mock_get_data.return_value = json.loads(sample_models_json.read_text())

        # Mock httpx to simulate remote failure
        with patch("httpx.Client") as mock_client:
            mock_response = Mock()
            mock_response.raise_for_status.side_effect = httpx.HTTPError("Failed")
            mock_client.return_value.get.return_value = mock_response

            registry = CapabilityRegistry(offline_mode=False)
            models = registry.get_models()

            # Should fall back to packaged data
            assert len(models) == 1
            assert models[0].model_id == "o3-mini"


def test_cache_behavior(sample_models_json):
    """Test caching of remote data."""
    with patch("llm_registry.registry.CapabilityRegistry._get_packaged_data") as mock_get_data:
        mock_get_data.return_value = json.loads(sample_models_json.read_text())

        # Mock httpx to count calls
        with patch("httpx.Client") as mock_client:
            mock_response = Mock()
            mock_response.raise_for_status.return_value = None
            mock_response.json.return_value = json.loads(sample_models_json.read_text())
            mock_client.return_value.get.return_value = mock_response

            registry = CapabilityRegistry(offline_mode=False, cache_ttl=10)

            # First call should hit remote
            registry.get_models()
            assert mock_client.return_value.get.call_count == 1

            # Second call should use cache
            registry.get_models()
            assert mock_client.return_value.get.call_count == 1
