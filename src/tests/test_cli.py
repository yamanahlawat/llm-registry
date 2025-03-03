"""Tests for the CLI module."""

from pathlib import Path
from unittest.mock import patch

import pytest
from llm_registry.cli import app
from llm_registry.models import ModelCapabilities, Provider, TokenCost
from typer.testing import CliRunner


@pytest.fixture
def runner():
    """Create a CLI runner."""
    return CliRunner()


@pytest.fixture
def sample_model():
    """Create a sample model capability."""
    return ModelCapabilities(
        model_id="test-model",
        provider=Provider.OPENAI,
        model_family="test",
        supports_streaming=True,
        supports_tools=True,
        supports_vision=False,
        supports_json_mode=True,
        supports_system_prompt=True,
        token_costs=TokenCost(input_cost=0.1, output_cost=0.2, context_window=1000, training_cutoff="2024-01"),
    )


def test_list_models(runner, sample_model):
    """Test listing models command."""
    with patch("llm_registry.cli.CapabilityRegistry") as mock_registry:
        mock_registry.return_value.get_models.return_value = [sample_model]
        result = runner.invoke(app, ["list"])
        assert result.exit_code == 0
        assert "test-model" in result.stdout
        assert "openai" in result.stdout
        assert "$0.1/1M" in result.stdout  # Input cost
        assert "$0.2/1M" in result.stdout  # Output cost


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


def test_add_model(runner, tmp_path):
    """Test adding a new model."""
    with patch("llm_registry.cli.CapabilityRepository") as mock_repo:
        # Mock get_model_capabilities to return None (model doesn't exist)
        mock_repo.return_value.get_model_capabilities.return_value = None
        mock_repo.return_value.save_model_capabilities.return_value = Path("test.json")
        result = runner.invoke(
            app,
            [
                "add",
                "test-model",
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
                "--streaming",
                "--tools",
                "--json-mode",
                "--system-prompt",
            ],
        )
        assert result.exit_code == 0
        assert "Model added successfully" in result.stdout


def test_delete_model(runner):
    """Test deleting a model."""
    with patch("llm_registry.cli.CapabilityRepository") as mock_repo:
        # Test successful deletion
        mock_repo.return_value.delete_model.return_value = True
        result = runner.invoke(app, ["delete", "test-model", "--provider", "openai", "-f"])
        assert result.exit_code == 0
        assert "deleted" in result.stdout

        # Test failed deletion
        mock_repo.return_value.delete_model.return_value = False
        result = runner.invoke(app, ["delete", "non-existent", "--provider", "openai", "-f"])
        assert result.exit_code == 1
        assert "Failed to delete model" in result.stdout
