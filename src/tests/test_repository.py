"""
Tests for the CapabilityRepository class.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from llm_registry import CapabilityRepository
from llm_registry.models import ApiParams, Features, ModelCapabilities, Provider, TokenCost


@pytest.fixture
def mock_user_models():
    return {
        "models": {
            "existing-model": {
                "providers": ["openai"],
                "model_family": "test-family",
                "api_params": {"stream": True},
                "features": {"vision": False, "tools": True, "json_mode": True, "system_prompt": True},
                "token_costs": {
                    "input_cost": 1.0,
                    "output_cost": 2.0,
                    "context_window": 8000,
                    "training_cutoff": "2023-12",
                },
            }
        }
    }


@pytest.fixture
def sample_model_capabilities():
    return ModelCapabilities(
        model_id="test-model",
        providers=[Provider.OPENAI],
        model_family="test-family",
        api_params=ApiParams(stream=True),
        features=Features(tools=True, json_mode=True, system_prompt=True),
        token_costs=TokenCost(input_cost=1.0, output_cost=2.0, context_window=8000, training_cutoff="2023-12"),
    )


@pytest.fixture
def repository(mock_user_models):
    with patch("llm_registry.repository.get_user_data_dir", return_value=Path("/mock/data/dir")):
        with patch("llm_registry.repository.load_user_models", return_value=mock_user_models):
            with patch("llm_registry.repository.save_user_models"):
                # Mock Path.mkdir to prevent actual filesystem operations
                with patch("pathlib.Path.mkdir"):
                    yield CapabilityRepository()


def test_init_creates_data_dir():
    """Test that initialization creates the data directory if it doesn't exist."""
    mock_path = MagicMock(spec=Path)
    with patch("llm_registry.repository.get_user_data_dir", return_value=mock_path):
        with patch("llm_registry.repository.load_user_models"):
            with patch("pathlib.Path.mkdir"):  # Prevent any actual directory creation
                CapabilityRepository()
                mock_path.mkdir.assert_called_once_with(parents=True, exist_ok=True)


def test_save_model_capabilities(repository, sample_model_capabilities):
    """Test saving model capabilities."""
    # Create a fake Path object for the models file
    mock_file_path = MagicMock(spec=Path)

    # Mock get_user_models_file to return our fake path
    with patch("llm_registry.repository.get_user_models_file", return_value=mock_file_path):
        with patch("llm_registry.repository.save_user_models") as mock_save:
            result = repository.save_model_capabilities(sample_model_capabilities)

            # Check that save_user_models was called with updated data
            called_data = mock_save.call_args[0][0]
            assert "test-model" in called_data["models"]
            assert called_data["models"]["test-model"]["model_family"] == "test-family"
            assert called_data["models"]["test-model"]["providers"] == ["openai"]

            # Verify the return value is the mocked path
            assert result == mock_file_path


def test_get_model_capabilities_existing(repository):
    """Test retrieving existing model capabilities."""
    model = repository.get_model_capabilities(Provider.OPENAI, "existing-model")
    assert model is not None
    assert model.model_id == "existing-model"
    assert model.token_costs.input_cost == 1.0
    assert model.features.tools is True


def test_get_model_capabilities_wrong_provider(repository):
    """Test retrieving model with wrong provider."""
    model = repository.get_model_capabilities(Provider.ANTHROPIC, "existing-model")
    assert model is None


def test_get_model_capabilities_nonexistent(repository):
    """Test retrieving non-existent model."""
    model = repository.get_model_capabilities(Provider.OPENAI, "nonexistent-model")
    assert model is None


def test_list_models_no_filter(repository):
    """Test listing all models."""
    models = repository.list_models()
    assert len(models) == 1
    assert models[0].model_id == "existing-model"


def test_list_models_with_provider_filter(repository):
    """Test listing models filtered by provider."""
    models = repository.list_models(provider=Provider.OPENAI)
    assert len(models) == 1
    assert models[0].model_id == "existing-model"

    models = repository.list_models(provider=Provider.ANTHROPIC)
    assert len(models) == 0


def test_delete_model_existing(repository):
    """Test deleting an existing model."""
    with patch("llm_registry.repository.save_user_models") as mock_save:
        result = repository.delete_model(Provider.OPENAI, "existing-model")
        assert result is True
        # Verify that model was removed from user models
        called_data = mock_save.call_args[0][0]
        assert "existing-model" not in called_data["models"]


def test_delete_model_wrong_provider(repository):
    """Test deleting model with wrong provider."""
    with patch("llm_registry.repository.save_user_models") as mock_save:
        result = repository.delete_model(Provider.ANTHROPIC, "existing-model")
        assert result is False
        mock_save.assert_not_called()


def test_delete_model_nonexistent(repository):
    """Test deleting non-existent model."""
    with patch("llm_registry.repository.save_user_models") as mock_save:
        result = repository.delete_model(Provider.OPENAI, "nonexistent-model")
        assert result is False
        mock_save.assert_not_called()


def test_delete_model_multi_provider(repository):
    """Test removing one provider from a model with multiple providers."""
    # Modify the mock data to include multiple providers
    repository._user_models["models"]["existing-model"]["providers"] = ["openai", "azure"]

    with patch("llm_registry.repository.save_user_models") as mock_save:
        result = repository.delete_model(Provider.OPENAI, "existing-model")
        assert result is True

        # Verify that only the provider was removed, not the entire model
        called_data = mock_save.call_args[0][0]
        assert "existing-model" in called_data["models"]
        assert called_data["models"]["existing-model"]["providers"] == ["azure"]
