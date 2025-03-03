"""Tests for the CLI module."""

import json
from pathlib import Path
from unittest.mock import mock_open, patch

import pytest
from typer.testing import CliRunner

from llm_registry.cli import app
from llm_registry.models import ApiParams, Features, ModelCapabilities, Provider, TokenCost


@pytest.fixture
def runner():
    """Create a CLI runner."""
    return CliRunner()


@pytest.fixture
def sample_model():
    """Create a sample model capability."""
    return ModelCapabilities(
        model_id="test-model",
        providers=[Provider.OPENAI],
        model_family="test",
        api_params=ApiParams(
            max_tokens=True,
            temperature=True,
            top_p=True,
            frequency_penalty=False,
            presence_penalty=False,
            stop=True,
            n=True,
            stream=True,
        ),
        features=Features(
            tools=True,
            vision=False,
            json_mode=True,
            system_prompt=True,
        ),
        token_costs=TokenCost(
            input_cost=0.1,
            output_cost=0.2,
            cache_input_cost=0.05,
            cache_output_cost=0.1,
            context_window=1000,
            training_cutoff="2024-01",
        ),
    )


@pytest.fixture
def mock_models_file(sample_model):
    """Create a mock models.json file."""
    return {
        "models": {
            "test-model": sample_model.model_dump(),
        }
    }


def test_list_models(runner, sample_model):
    """Test listing models command."""
    with patch("llm_registry.cli.CapabilityRegistry") as mock_registry:
        mock_registry.return_value.get_models.return_value = [sample_model]
        result = runner.invoke(app, ["list"])
        assert result.exit_code == 0
        # Check basic info
        assert "test-model" in result.stdout
        assert "openai" in result.stdout
        assert "test" in result.stdout  # model family
        # Check costs (just basic presence)
        assert "$0.1" in result.stdout  # Input cost
        assert "$0.2" in result.stdout  # Output cost


def test_list_models_by_provider(runner, sample_model):
    """Test listing models filtered by provider."""
    with patch("llm_registry.cli.CapabilityRegistry") as mock_registry:
        mock_registry.return_value.get_models.return_value = [sample_model]
        result = runner.invoke(app, ["list", "--provider", "openai"])
        assert result.exit_code == 0
        assert "test-model" in result.stdout

        # Test non-existent provider
        mock_registry.return_value.get_models.return_value = []
        result = runner.invoke(app, ["list", "--provider", "anthropic"])
        assert result.exit_code == 0
        assert "No models found" in result.stdout


def test_add_model(runner, mock_models_file):
    """Test adding a new model."""
    mock_file = mock_open(read_data=json.dumps(mock_models_file))
    with patch("builtins.open", mock_file), patch("llm_registry.cli.get_models_file") as mock_path:
        mock_path.return_value = Path("test.json")
        result = runner.invoke(
            app,
            [
                "add",
                "new-model",
                "--provider",
                "openai",
                "--model-family",
                "test",
                "--input-cost",
                "0.1",
                "--output-cost",
                "0.2",
                "--context-window",
                "1000",
                "--training-cutoff",
                "2024-01",
                "--stream",
                "--tools",
                "--json-mode",
                "--system-prompt",
            ],
        )
        assert result.exit_code == 0
        assert "Added new model 'new-model'" in result.stdout


def test_add_existing_model_provider(runner, mock_models_file):
    """Test adding a provider to an existing model."""
    mock_file = mock_open(read_data=json.dumps(mock_models_file))
    with patch("builtins.open", mock_file), patch("llm_registry.cli.get_models_file") as mock_path:
        mock_path.return_value = Path("test.json")
        result = runner.invoke(
            app,
            [
                "add",
                "test-model",
                "--provider",
                "anthropic",
            ],
        )
        assert result.exit_code == 0
        assert "Added provider 'anthropic' to existing model 'test-model'" in result.stdout


def test_delete_model(runner, mock_models_file):
    """Test deleting a model."""
    mock_file = mock_open(read_data=json.dumps(mock_models_file))
    with patch("builtins.open", mock_file), patch("llm_registry.cli.get_models_file") as mock_path:
        mock_path.return_value = Path("test.json")

        # Test deleting entire model with force flag
        result = runner.invoke(app, ["delete", "test-model", "-f"])
        assert result.exit_code == 0
        assert "Model 'test-model' deleted" in result.stdout

        # Test deleting non-existent model
        result = runner.invoke(app, ["delete", "non-existent", "-f"])
        assert result.exit_code == 1
        assert "not found" in result.stdout


def test_delete_provider(runner, mock_models_file):
    """Test removing a provider from a model."""
    mock_file = mock_open(read_data=json.dumps(mock_models_file))
    with patch("builtins.open", mock_file), patch("llm_registry.cli.get_models_file") as mock_path:
        mock_path.return_value = Path("test.json")

        # Test removing existing provider
        result = runner.invoke(app, ["delete", "test-model", "--provider", "openai", "-f"])
        assert result.exit_code == 0
        assert "deleted (no providers remaining)" in result.stdout

        # Test removing non-existent provider
        result = runner.invoke(app, ["delete", "test-model", "--provider", "anthropic", "-f"])
        assert result.exit_code == 1
        assert "not found" in result.stdout


def test_update_model(runner, mock_models_file):
    """Test updating a model's capabilities."""
    mock_file = mock_open(read_data=json.dumps(mock_models_file))
    with patch("builtins.open", mock_file), patch("llm_registry.cli.get_models_file") as mock_path:
        mock_path.return_value = Path("test.json")

        # Test updating various fields
        result = runner.invoke(
            app,
            [
                "update",
                "test-model",
                "--model-family",
                "new-family",
                "--input-cost",
                "0.15",
                "--cache-input-cost",
                "0.075",
                "--context-window",
                "2000",
                "--vision",
            ],
        )
        assert result.exit_code == 0
        assert "Updated model 'test-model'" in result.stdout

        # Test updating non-existent model
        result = runner.invoke(app, ["update", "non-existent"])
        assert result.exit_code == 1
        assert "not found" in result.stdout


def test_update_provider_specific(runner, mock_models_file):
    """Test updating provider-specific capabilities."""
    mock_file = mock_open(read_data=json.dumps(mock_models_file))
    with patch("builtins.open", mock_file), patch("llm_registry.cli.get_models_file") as mock_path:
        mock_path.return_value = Path("test.json")

        # Test updating with non-existent provider
        result = runner.invoke(
            app,
            [
                "update",
                "test-model",
                "--provider",
                "anthropic",
                "--input-cost",
                "0.15",
            ],
        )
        assert result.exit_code == 1
        assert "not found" in result.stdout
