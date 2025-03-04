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
def mock_package_models(sample_model):
    """Create mock package models."""
    return {
        "models": {
            "package-model": {
                "providers": ["openai"],
                "model_family": "package",
                "api_params": {
                    "max_tokens": True,
                    "temperature": True,
                    "stream": True
                },
                "features": {
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


@pytest.fixture
def mock_user_models(sample_model):
    """Create mock user models."""
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
        assert "test-model" in result.stdout
        assert "openai" in result.stdout
        assert "test" in result.stdout  # model family
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


def test_list_models_by_source(runner, sample_model):
    """Test listing models filtered by source."""
    with patch("llm_registry.cli.CapabilityRegistry") as mock_registry, \
         patch("llm_registry.cli.load_user_models") as mock_user, \
         patch("llm_registry.cli.load_package_models") as mock_package:
        
        # Setup mocks
        mock_registry.return_value.get_models.return_value = [sample_model]
        mock_user.return_value = {"models": {"test-model": {}}}
        mock_package.return_value = {"models": {"package-model": {}}}

        # Test user-only
        result = runner.invoke(app, ["list", "--user-only"])
        assert result.exit_code == 0
        assert "test-model" in result.stdout
        assert "package-model" not in result.stdout

        # Test package-only
        result = runner.invoke(app, ["list", "--package-only"])
        assert result.exit_code == 0
        assert "package-model" in result.stdout
        assert "test-model" not in result.stdout

        # Test both flags (should error)
        result = runner.invoke(app, ["list", "--user-only", "--package-only"])
        assert result.exit_code == 1
        assert "Cannot specify both" in result.stdout


def test_get_model(runner, sample_model):
    """Test getting model details."""
    with patch("llm_registry.cli.CapabilityRegistry") as mock_registry, \
         patch("llm_registry.cli.load_user_models") as mock_user:
        
        mock_registry.return_value.get_model.return_value = sample_model
        mock_user.return_value = {"models": {"test-model": {}}}

        # Test normal output
        result = runner.invoke(app, ["get", "test-model"])
        assert result.exit_code == 0
        assert "Model: test-model" in result.stdout
        assert "Source: User" in result.stdout
        assert "openai" in result.stdout
        assert "$0.1" in result.stdout
        assert "" in result.stdout  # For boolean features

        # Test JSON output
        result = runner.invoke(app, ["get", "test-model", "--json"])
        assert result.exit_code == 0
        data = json.loads(result.stdout)
        assert data["model_id"] == "test-model"

        # Test non-existent model
        mock_registry.return_value.get_model.side_effect = KeyError("not found")
        result = runner.invoke(app, ["get", "non-existent"])
        assert result.exit_code == 1
        assert "not found" in result.stdout


def test_add_model(runner, mock_user_models, mock_package_models):
    """Test adding a new model."""
    mock_file = mock_open(read_data=json.dumps(mock_user_models))
    with patch("builtins.open", mock_file), \
         patch("llm_registry.cli.get_models_file") as mock_path, \
         patch("llm_registry.cli.is_package_model") as mock_is_package:
        
        mock_path.return_value = Path("test.json")
        mock_is_package.return_value = False

        # Test adding new model
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

        # Test adding package model (should fail)
        mock_is_package.return_value = True
        result = runner.invoke(
            app,
            [
                "add",
                "package-model",
                "--provider",
                "openai",
            ],
        )
        assert result.exit_code == 1
        assert "Package models are read-only" in result.stdout


def test_delete_model(runner, mock_user_models):
    """Test deleting a model."""
    mock_file = mock_open(read_data=json.dumps(mock_user_models))
    with patch("builtins.open", mock_file), \
         patch("llm_registry.cli.get_models_file") as mock_path, \
         patch("llm_registry.cli.is_package_model") as mock_is_package:
        
        mock_path.return_value = Path("test.json")
        mock_is_package.return_value = False

        # Test deleting user model
        result = runner.invoke(app, ["delete", "test-model", "-f"])
        assert result.exit_code == 0
        assert "Deleted model 'test-model'" in result.stdout

        # Test deleting package model (should fail)
        mock_is_package.return_value = True
        result = runner.invoke(app, ["delete", "package-model", "-f"])
        assert result.exit_code == 1
        assert "Package models are read-only" in result.stdout


def test_update_model(runner, mock_user_models):
    """Test updating a model."""
    mock_file = mock_open(read_data=json.dumps(mock_user_models))
    with patch("builtins.open", mock_file), \
         patch("llm_registry.cli.get_models_file") as mock_path, \
         patch("llm_registry.cli.is_package_model") as mock_is_package:
        
        mock_path.return_value = Path("test.json")
        mock_is_package.return_value = False

        # Test updating user model
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

        # Test updating package model (should fail)
        mock_is_package.return_value = True
        result = runner.invoke(
            app,
            [
                "update",
                "package-model",
                "--model-family",
                "new-family",
            ],
        )
        assert result.exit_code == 1
        assert "Package models are read-only" in result.stdout
