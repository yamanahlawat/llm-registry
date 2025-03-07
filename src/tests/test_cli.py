"""
Tests for CLI commands.
"""

import json
from unittest.mock import MagicMock, patch

import pytest
import typer
from typer.testing import CliRunner

from llm_registry.cli import app, get_model_data, validate_not_package_model, validate_provider
from llm_registry.exceptions import ModelNotFoundError
from llm_registry.models import ModelCapabilities, Provider


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
            }
        }
    }


@pytest.fixture
def mock_user_models():
    return {
        "models": {
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
            }
        }
    }


@pytest.fixture
def runner():
    return CliRunner()


def test_list_command(runner, mock_package_models, mock_user_models):
    """Test the 'list' command."""
    with patch("llm_registry.cli.load_package_models", return_value=mock_package_models):
        with patch("llm_registry.cli.load_user_models", return_value=mock_user_models):
            # Also mock the CapabilityRegistry to ensure it uses our mocked data
            with patch("llm_registry.cli.CapabilityRegistry") as mock_registry_class:
                # Setup the mock registry instance
                mock_registry = MagicMock()
                mock_registry_class.return_value = mock_registry

                # Create model instances from our mock data
                gpt4o_model = ModelCapabilities.model_validate(
                    {**mock_package_models["models"]["gpt-4o"], "model_id": "gpt-4o"}
                )
                custom_model = ModelCapabilities.model_validate(
                    {**mock_user_models["models"]["custom-model"], "model_id": "custom-model"}
                )

                # Configure the mock registry to return our models
                mock_registry.get_models.return_value = [gpt4o_model, custom_model]

                result = runner.invoke(app, ["list"])
                assert result.exit_code == 0
                # Check that both models are listed
                assert "gpt-4o" in result.stdout
                assert "custom-model" in result.stdout
                # Check column headers
                assert "Model ID" in result.stdout
                assert "Providers" in result.stdout
                assert "Family" in result.stdout


def test_list_command_with_provider_filter(runner, mock_package_models, mock_user_models):
    """Test the 'list' command with provider filter."""
    with patch("llm_registry.cli.load_package_models", return_value=mock_package_models):
        with patch("llm_registry.cli.load_user_models", return_value=mock_user_models):
            with patch("llm_registry.cli.CapabilityRegistry") as mock_registry_class:
                mock_registry = MagicMock()
                mock_registry_class.return_value = mock_registry

                # Create model instances
                gpt4o_model = ModelCapabilities.model_validate(
                    {**mock_package_models["models"]["gpt-4o"], "model_id": "gpt-4o"}
                )
                custom_model = ModelCapabilities.model_validate(
                    {**mock_user_models["models"]["custom-model"], "model_id": "custom-model"}
                )

                # Configure for openai filter
                mock_registry.get_models.side_effect = lambda provider=None: {
                    Provider.OPENAI: [gpt4o_model, custom_model],
                    Provider.ANTHROPIC: [],
                }.get(provider, [gpt4o_model, custom_model])

                # Test openai filter
                result = runner.invoke(app, ["list", "--provider", "openai"])
                assert result.exit_code == 0
                assert "gpt-4o" in result.stdout
                assert "custom-model" in result.stdout

                # Test anthropic filter
                result = runner.invoke(app, ["list", "--provider", "anthropic"])
                assert result.exit_code == 0
                assert "No models found" in result.stdout


def test_list_command_user_only(runner, mock_package_models, mock_user_models):
    """Test the 'list' command with --user-only."""
    with patch("llm_registry.cli.load_package_models", return_value=mock_package_models):
        with patch("llm_registry.cli.load_user_models", return_value=mock_user_models):
            result = runner.invoke(app, ["list", "--user-only"])
            assert result.exit_code == 0
            assert "custom-model" in result.stdout
            assert "gpt-4o" not in result.stdout


def test_list_command_package_only(runner, mock_package_models, mock_user_models):
    """Test the 'list' command with --package-only."""
    with patch("llm_registry.cli.load_package_models", return_value=mock_package_models):
        with patch("llm_registry.cli.load_user_models", return_value=mock_user_models):
            result = runner.invoke(app, ["list", "--package-only"])
            assert result.exit_code == 0
            assert "gpt-4o" in result.stdout
            assert "custom-model" not in result.stdout


def test_get_command(runner, mock_package_models, mock_user_models):
    """Test the 'get' command."""
    with patch("llm_registry.cli.load_package_models", return_value=mock_package_models):
        with patch("llm_registry.cli.load_user_models", return_value=mock_user_models):
            with patch("llm_registry.cli.CapabilityRegistry") as mock_registry_class:
                mock_registry = MagicMock()
                mock_registry_class.return_value = mock_registry

                # Create a model instance
                gpt4o_model = ModelCapabilities.model_validate(
                    {**mock_package_models["models"]["gpt-4o"], "model_id": "gpt-4o"}
                )

                # Configure get_model to return our model
                mock_registry.get_model.return_value = gpt4o_model

                # Test normal output
                result = runner.invoke(app, ["get", "gpt-4o"])
                assert result.exit_code == 0

                # Look for key parts of the output without requiring exact formatting
                assert "Model: gpt-4o" in result.stdout
                assert "Source" in result.stdout
                assert "Package" in result.stdout
                assert "Model Family" in result.stdout
                assert "gpt-4" in result.stdout

                # Test JSON output
                result = runner.invoke(app, ["get", "gpt-4o", "--json"])
                assert result.exit_code == 0
                # Ensure output is valid JSON
                model_data = json.loads(result.stdout)
                assert model_data["model_id"] == "gpt-4o"


def test_get_command_nonexistent_model(runner, mock_package_models, mock_user_models):
    """Test the 'get' command with a non-existent model."""
    with patch("llm_registry.cli.load_package_models", return_value=mock_package_models):
        with patch("llm_registry.cli.load_user_models", return_value=mock_user_models):
            result = runner.invoke(app, ["get", "nonexistent-model"])
            assert result.exit_code == 1
            assert "Error: Model 'nonexistent-model' not found in registry" in result.stdout


def test_add_command(runner, mock_user_models):
    """Test the 'add' command."""
    with patch("llm_registry.cli.is_package_model", return_value=False):
        with patch("llm_registry.cli.load_user_models", return_value=mock_user_models):
            with patch("llm_registry.cli.save_user_models") as mock_save:
                result = runner.invoke(
                    app,
                    [
                        "add",
                        "new-model",
                        "--provider",
                        "openai",
                        "--model-family",
                        "new-family",
                        "--input-cost",
                        "1.5",
                        "--output-cost",
                        "3.0",
                        "--context-window",
                        "10000",
                        "--training-cutoff",
                        "2024-01",
                        "--stream",
                        "--tools",
                        "--json-mode",
                    ],
                )
                assert result.exit_code == 0
                assert "Added model 'openai/new-model' to user registry" in result.stdout

                # Check that save_user_models was called with the correct data
                called_data = mock_save.call_args[0][0]
                assert "new-model" in called_data["models"]
                assert called_data["models"]["new-model"]["providers"] == ["openai"]
                assert called_data["models"]["new-model"]["model_family"] == "new-family"
                assert called_data["models"]["new-model"]["token_costs"]["input_cost"] == 1.5
                assert called_data["models"]["new-model"]["token_costs"]["output_cost"] == 3.0
                assert called_data["models"]["new-model"]["api_params"]["stream"] is True
                assert called_data["models"]["new-model"]["features"]["tools"] is True
                assert called_data["models"]["new-model"]["features"]["json_mode"] is True


def test_add_command_package_model(runner):
    """Test the 'add' command for a package model (should fail)."""
    with patch("llm_registry.cli.is_package_model", return_value=True):
        result = runner.invoke(app, ["add", "gpt-4o", "--provider", "openai"])
        assert result.exit_code == 1
        assert "Cannot add model 'gpt-4o'. Package models are read-only" in result.stdout


def test_add_command_existing_model_new_provider(runner, mock_user_models):
    """Test adding a new provider to an existing model."""
    with patch("llm_registry.cli.is_package_model", return_value=False):
        with patch("llm_registry.cli.load_user_models", return_value=mock_user_models):
            with patch("llm_registry.cli.save_user_models") as mock_save:
                result = runner.invoke(app, ["add", "custom-model", "--provider", "anthropic"])
                assert result.exit_code == 0
                assert "Added provider 'anthropic' to existing model 'custom-model'" in result.stdout

                # Check that provider was added to existing model
                called_data = mock_save.call_args[0][0]
                assert "custom-model" in called_data["models"]
                assert "anthropic" in called_data["models"]["custom-model"]["providers"]
                assert "openai" in called_data["models"]["custom-model"]["providers"]


def test_update_command(runner, mock_user_models):
    """Test the 'update' command."""
    with patch("llm_registry.cli.is_package_model", return_value=False):
        with patch("llm_registry.cli.load_user_models", return_value=mock_user_models):
            with patch("llm_registry.cli.save_user_models") as mock_save:
                result = runner.invoke(
                    app,
                    [
                        "update",
                        "custom-model",
                        "--model-family",
                        "updated-family",
                        "--input-cost",
                        "2.0",
                        "--stream",
                        "--tools",
                    ],
                )
                assert result.exit_code == 0
                assert "Updated model 'custom-model'" in result.stdout

                # Check that model was updated correctly
                called_data = mock_save.call_args[0][0]
                assert called_data["models"]["custom-model"]["model_family"] == "updated-family"
                assert called_data["models"]["custom-model"]["token_costs"]["input_cost"] == 2.0
                assert called_data["models"]["custom-model"]["api_params"]["stream"] is True
                assert called_data["models"]["custom-model"]["features"]["tools"] is True


def test_update_command_nonexistent_model(runner, mock_user_models):
    """Test updating a non-existent model."""
    with patch("llm_registry.cli.is_package_model", return_value=False):
        with patch("llm_registry.cli.load_user_models", return_value=mock_user_models):
            result = runner.invoke(app, ["update", "nonexistent-model", "--model-family", "new-family"])
            assert result.exit_code == 1
            assert "Error: Model 'nonexistent-model' not found in user registry" in result.stdout


def test_update_command_package_model(runner):
    """Test updating a package model (should fail)."""
    with patch("llm_registry.cli.is_package_model", return_value=True):
        result = runner.invoke(app, ["update", "gpt-4o", "--model-family", "new-family"])
        assert result.exit_code == 1
        assert "Cannot update model 'gpt-4o'. Package models are read-only" in result.stdout


def test_delete_command(runner, mock_user_models):
    """Test the 'delete' command with force flag."""
    with patch("llm_registry.cli.is_package_model", return_value=False):
        with patch("llm_registry.cli.load_user_models", return_value=mock_user_models):
            with patch("llm_registry.cli.save_user_models") as mock_save:
                result = runner.invoke(app, ["delete", "custom-model", "--force"])
                assert result.exit_code == 0
                assert "Deleted model 'custom-model' from user registry" in result.stdout

                # Check that model was removed
                called_data = mock_save.call_args[0][0]
                assert "custom-model" not in called_data["models"]


def test_delete_command_with_provider(runner, mock_user_models):
    """Test the 'delete' command with a provider specified."""
    # Modify mock data to have multiple providers
    mock_user_models["models"]["custom-model"]["providers"] = ["openai", "anthropic"]

    with patch("llm_registry.cli.is_package_model", return_value=False):
        with patch("llm_registry.cli.load_user_models", return_value=mock_user_models):
            with patch("llm_registry.cli.save_user_models") as mock_save:
                result = runner.invoke(app, ["delete", "custom-model", "--provider", "openai", "--force"])
                assert result.exit_code == 0
                assert "Removed provider 'openai' from model 'custom-model'" in result.stdout

                # Check that only the provider was removed
                called_data = mock_save.call_args[0][0]
                assert "custom-model" in called_data["models"]
                assert called_data["models"]["custom-model"]["providers"] == ["anthropic"]


def test_delete_command_provider_last_provider(runner, mock_user_models):
    """Test removing the last provider from a model (should delete model)."""
    with patch("llm_registry.cli.is_package_model", return_value=False):
        with patch("llm_registry.cli.load_user_models", return_value=mock_user_models):
            with patch("llm_registry.cli.save_user_models") as mock_save:
                result = runner.invoke(app, ["delete", "custom-model", "--provider", "openai", "--force"])
                assert result.exit_code == 0
                assert "Deleted model 'custom-model' from user registry" in result.stdout

                # Check that model was completely removed
                called_data = mock_save.call_args[0][0]
                assert "custom-model" not in called_data["models"]


def test_delete_command_nonexistent_model(runner, mock_user_models):
    """Test deleting a non-existent model."""
    with patch("llm_registry.cli.is_package_model", return_value=False):
        with patch("llm_registry.cli.load_user_models", return_value=mock_user_models):
            result = runner.invoke(app, ["delete", "nonexistent-model", "--force"])
            assert result.exit_code == 1
            assert "Error: Model 'nonexistent-model' not found in user registry" in result.stdout


def test_delete_command_package_model(runner):
    """Test deleting a package model (should fail)."""
    with patch("llm_registry.cli.is_package_model", return_value=True):
        result = runner.invoke(app, ["delete", "gpt-4o", "--force"])
        assert result.exit_code == 1
        assert "Cannot delete model 'gpt-4o'. Package models are read-only" in result.stdout


def test_get_model_data_not_found():
    """Test get_model_data when model not found."""
    test_data = {"models": {"existing-model": {}}}

    with pytest.raises(typer.Exit) as excinfo:
        with patch("llm_registry.cli.console.print") as mock_print:
            get_model_data("non-existent-model", test_data)

    assert excinfo.value.exit_code == 1
    mock_print.assert_called_once()
    assert "not found" in mock_print.call_args[0][0]


def test_validate_provider_invalid():
    """Test validate_provider with invalid provider."""
    model_data = {"providers": ["openai"]}

    with pytest.raises(typer.Exit) as excinfo:
        with patch("llm_registry.cli.console.print") as mock_print:
            validate_provider(model_data, Provider.ANTHROPIC, "test-model")

    assert excinfo.value.exit_code == 1
    mock_print.assert_called_once()
    assert "not found for model" in mock_print.call_args[0][0]


def test_validate_not_package_model():
    """Test validate_not_package_model with package model."""
    with patch("llm_registry.cli.is_package_model", return_value=True):
        with pytest.raises(typer.Exit) as excinfo:
            with patch("llm_registry.cli.console.print") as mock_print:
                validate_not_package_model("gpt-4o", "update")

        assert excinfo.value.exit_code == 1
        mock_print.assert_called_once()
        assert "Cannot update model" in mock_print.call_args[0][0]
        assert "Package models are read-only" in mock_print.call_args[0][0]


def test_list_command_conflicting_flags(runner):
    """Test 'list' command with conflicting flags."""
    result = runner.invoke(app, ["list", "--user-only", "--package-only"])
    assert result.exit_code == 1
    assert "Cannot specify both --user-only and --package-only" in result.stdout


def test_get_command_model_not_found(runner):
    """Test 'get' command with non-existent model."""
    with patch("llm_registry.cli.CapabilityRegistry") as mock_registry_class:
        mock_registry = MagicMock()
        mock_registry_class.return_value = mock_registry

        # Make get_model raise ModelNotFoundError
        mock_registry.get_model.side_effect = ModelNotFoundError("Model 'non-existent-model' not found in registry")

        result = runner.invoke(app, ["get", "non-existent-model"])
        assert result.exit_code == 1
        assert "Error: Model 'non-existent-model' not found in registry" in result.stdout


def test_add_command_existing_provider(runner):
    """Test adding a model with existing provider."""
    mock_user_models = {"models": {"existing-model": {"providers": ["openai"], "model_family": "test-family"}}}

    with patch("llm_registry.cli.is_package_model", return_value=False):
        with patch("llm_registry.cli.load_user_models", return_value=mock_user_models):
            with patch("llm_registry.cli.save_user_models"):
                result = runner.invoke(app, ["add", "existing-model", "--provider", "openai"])
                assert result.exit_code == 1
                assert "already exists" in result.stdout


def test_update_command_cache_params(runner):
    """Test updating model with cache parameters."""
    mock_user_models = {"models": {"custom-model": {"providers": ["openai"], "model_family": "test-family"}}}

    with patch("llm_registry.cli.is_package_model", return_value=False):
        with patch("llm_registry.cli.load_user_models", return_value=mock_user_models):
            with patch("llm_registry.cli.save_user_models") as mock_save:
                result = runner.invoke(
                    app, ["update", "custom-model", "--cache-input-cost", "0.5", "--cache-output-cost", "1.0"]
                )

                assert result.exit_code == 0
                assert "Updated model 'custom-model'" in result.stdout

                # Verify cache costs were added
                called_data = mock_save.call_args[0][0]
                assert "token_costs" in called_data["models"]["custom-model"]
                assert called_data["models"]["custom-model"]["token_costs"]["cache_input_cost"] == 0.5
                assert called_data["models"]["custom-model"]["token_costs"]["cache_output_cost"] == 1.0


def test_update_command_api_params(runner):
    """Test updating model API parameters."""
    mock_user_models = {"models": {"custom-model": {"providers": ["openai"], "model_family": "test-family"}}}

    with patch("llm_registry.cli.is_package_model", return_value=False):
        with patch("llm_registry.cli.load_user_models", return_value=mock_user_models):
            with patch("llm_registry.cli.save_user_models") as mock_save:
                result = runner.invoke(app, ["update", "custom-model", "--max-tokens", "--temperature", "--top-p"])

                assert result.exit_code == 0

                # Verify API params were added
                called_data = mock_save.call_args[0][0]
                assert "api_params" in called_data["models"]["custom-model"]
                assert called_data["models"]["custom-model"]["api_params"]["max_tokens"] is True
                assert called_data["models"]["custom-model"]["api_params"]["temperature"] is True
                assert called_data["models"]["custom-model"]["api_params"]["top_p"] is True


def test_update_command_features(runner):
    """Test updating model features."""
    mock_user_models = {"models": {"custom-model": {"providers": ["openai"], "model_family": "test-family"}}}

    with patch("llm_registry.cli.is_package_model", return_value=False):
        with patch("llm_registry.cli.load_user_models", return_value=mock_user_models):
            with patch("llm_registry.cli.save_user_models") as mock_save:
                result = runner.invoke(app, ["update", "custom-model", "--vision", "--json-mode", "--system-prompt"])

                assert result.exit_code == 0

                # Verify features were added
                called_data = mock_save.call_args[0][0]
                assert "features" in called_data["models"]["custom-model"]
                assert called_data["models"]["custom-model"]["features"]["vision"] is True
                assert called_data["models"]["custom-model"]["features"]["json_mode"] is True
                assert called_data["models"]["custom-model"]["features"]["system_prompt"] is True


def test_delete_command_provider_not_found(runner):
    """Test deleting a provider that doesn't exist for a model."""
    mock_user_models = {"models": {"custom-model": {"providers": ["openai"], "model_family": "test-family"}}}

    with patch("llm_registry.cli.is_package_model", return_value=False):
        with patch("llm_registry.cli.load_user_models", return_value=mock_user_models):
            result = runner.invoke(app, ["delete", "custom-model", "--provider", "anthropic", "--force"])

            assert result.exit_code == 1
            assert "not found for model" in result.stdout


def test_get_model_data_default_param():
    """Test get_model_data with default None data parameter."""
    test_model_data = {"test-key": "test-value"}
    test_data = {"models": {"test-model": test_model_data}}

    with patch("llm_registry.cli.load_user_models", return_value=test_data):
        result_data, result_model = get_model_data("test-model")

        # Verify the function loaded user models when data=None
        assert result_data == test_data
        assert result_model == test_model_data


def test_add_command_no_costs(runner):
    """Test add command with no cost parameters."""
    mock_user_models = {"models": {}}

    with patch("llm_registry.cli.is_package_model", return_value=False):
        with patch("llm_registry.cli.load_user_models", return_value=mock_user_models):
            with patch("llm_registry.cli.save_user_models") as mock_save:
                # Call add with minimal parameters (no costs)
                result = runner.invoke(
                    app,
                    [
                        "add",
                        "no-cost-model",
                        "--provider",
                        "openai",
                    ],
                )

                assert result.exit_code == 0
                assert "Added model" in result.stdout

                # Verify token_costs is None or not set
                called_data = mock_save.call_args[0][0]
                assert "no-cost-model" in called_data["models"]
                # Either token_costs should be missing or it should be None
                assert (
                    "token_costs" not in called_data["models"]["no-cost-model"]
                    or called_data["models"]["no-cost-model"]["token_costs"] is None
                )


def test_delete_command_without_provider(runner):
    """Test delete command without specifying a provider."""
    mock_user_models = {"models": {"test-model": {"providers": ["openai"], "model_family": "test-family"}}}

    with patch("llm_registry.cli.is_package_model", return_value=False):
        with patch("llm_registry.cli.load_user_models", return_value=mock_user_models):
            with patch("llm_registry.cli.save_user_models") as mock_save:
                # Use force flag to bypass confirmation
                result = runner.invoke(app, ["delete", "test-model", "--force"])

                assert result.exit_code == 0
                assert "Deleted model 'test-model'" in result.stdout

                # Verify model was deleted
                called_data = mock_save.call_args[0][0]
                assert "test-model" not in called_data["models"]


def test_delete_command_with_confirmation(runner):
    """Test delete command with interactive confirmation."""
    mock_user_models = {"models": {"test-model": {"providers": ["openai"], "model_family": "test-family"}}}

    with patch("llm_registry.cli.is_package_model", return_value=False):
        with patch("llm_registry.cli.load_user_models", return_value=mock_user_models):
            with patch("llm_registry.cli.save_user_models") as mock_save:
                # Simulate user typing 'y' for confirmation
                result = runner.invoke(app, ["delete", "test-model"], input="y\n")

                assert result.exit_code == 0
                assert "Delete model 'test-model'" in result.stdout
                assert "Deleted model 'test-model'" in result.stdout

                # Verify model was deleted
                called_data = mock_save.call_args[0][0]
                assert "test-model" not in called_data["models"]


def test_delete_command_confirmation_cancel(runner):
    """Test delete command with user canceling confirmation."""
    mock_user_models = {"models": {"test-model": {"providers": ["openai"], "model_family": "test-family"}}}

    with patch("llm_registry.cli.is_package_model", return_value=False):
        with patch("llm_registry.cli.load_user_models", return_value=mock_user_models):
            with patch("llm_registry.cli.save_user_models") as mock_save:
                # Simulate user typing 'n' to cancel
                result = runner.invoke(app, ["delete", "test-model"], input="n\n")

                # Should exit without error but without deleting
                assert result.exit_code == 0
                assert "Delete model 'test-model'" in result.stdout

                # Save should not have been called
                mock_save.assert_not_called()


def test_delete_provider_with_confirmation(runner):
    """Test delete command with provider and interactive confirmation."""
    mock_user_models = {"models": {"test-model": {"providers": ["openai", "anthropic"], "model_family": "test-family"}}}

    with patch("llm_registry.cli.is_package_model", return_value=False):
        with patch("llm_registry.cli.load_user_models", return_value=mock_user_models):
            with patch("llm_registry.cli.save_user_models") as mock_save:
                # Simulate user typing 'y' for confirmation
                result = runner.invoke(app, ["delete", "test-model", "--provider", "openai"], input="y\n")

                assert result.exit_code == 0
                assert "Remove provider 'openai' from model 'test-model'" in result.stdout
                assert "Removed provider 'openai'" in result.stdout

                # Verify provider was removed but model remains
                called_data = mock_save.call_args[0][0]
                assert "test-model" in called_data["models"]
                assert "openai" not in called_data["models"]["test-model"]["providers"]
                assert "anthropic" in called_data["models"]["test-model"]["providers"]


def test_get_command_json_output(runner):
    """Test the get command with JSON output flag."""

    with patch("llm_registry.cli.CapabilityRegistry") as mock_registry:
        # Setup a mock model and its JSON output
        mock_instance = mock_registry.return_value
        mock_model = MagicMock()
        mock_model.model_dump_json.return_value = '{"model_id": "test-model"}'
        mock_instance.get_model.return_value = mock_model

        result = runner.invoke(app, ["get", "test-model", "--json"])
        assert result.exit_code == 0

        # Parse the JSON from the output and compare objects instead of strings
        result_json = json.loads(result.stdout)
        expected_json = {"model_id": "test-model"}
        assert result_json == expected_json


def test_add_command_with_specific_features(runner):
    """Test add command with a specific set of features."""
    mock_user_models = {"models": {}}

    with patch("llm_registry.cli.is_package_model", return_value=False):
        with patch("llm_registry.cli.load_user_models", return_value=mock_user_models):
            with patch("llm_registry.cli.save_user_models") as mock_save:
                # Try with just vision feature and no other features or params
                result = runner.invoke(app, ["add", "vision-model", "--provider", "openai", "--vision"])

                assert result.exit_code == 0
                assert "Added model" in result.stdout

                # Verify vision was set - accept that token_costs might be None
                called_data = mock_save.call_args[0][0]
                assert "vision-model" in called_data["models"]
                assert called_data["models"]["vision-model"]["features"]["vision"] is True
                # Either token_costs is None or not set at all
                if "token_costs" in called_data["models"]["vision-model"]:
                    assert called_data["models"]["vision-model"]["token_costs"] is None


def test_update_command_empty_changes(runner):
    """Test update command with no actual changes."""
    mock_user_models = {"models": {"test-model": {"providers": ["openai"]}}}

    with patch("llm_registry.cli.is_package_model", return_value=False):
        with patch("llm_registry.cli.load_user_models", return_value=mock_user_models):
            with patch("llm_registry.cli.save_user_models") as mock_save:
                # Call update with no actual changes
                result = runner.invoke(app, ["update", "test-model"])

                assert result.exit_code == 0
                assert "Updated model 'test-model'" in result.stdout

                # Verify save was called with unchanged data
                called_data = mock_save.call_args[0][0]
                assert called_data["models"]["test-model"]["providers"] == ["openai"]


def test_delete_command_specific_provider_condition(runner):
    """Test a specific provider deletion condition to target line 379."""
    mock_user_models = {
        "models": {
            "multi-provider-model": {"providers": ["openai", "anthropic", "google"], "model_family": "test-family"}
        }
    }

    with patch("llm_registry.cli.is_package_model", return_value=False):
        with patch("llm_registry.cli.load_user_models", return_value=mock_user_models):
            with patch("llm_registry.cli.save_user_models") as mock_save:
                # Delete the middle provider
                result = runner.invoke(app, ["delete", "multi-provider-model", "--provider", "anthropic", "--force"])

                assert result.exit_code == 0
                assert "Removed provider" in result.stdout

                # Verify only anthropic was removed
                called_data = mock_save.call_args[0][0]
                assert "multi-provider-model" in called_data["models"]
                providers = called_data["models"]["multi-provider-model"]["providers"]
                assert "anthropic" not in providers
                assert "openai" in providers
                assert "google" in providers


def test_get_command_with_specific_token_costs(runner):
    """Test get command output with specific token cost structure (lines 183, 185)."""
    # Mock a model with cache costs explicitly set to None
    mock_model_data = {
        "model_id": "test-model",
        "providers": ["openai"],
        "model_family": "test-family",
        "token_costs": {
            "input_cost": 1.0,
            "output_cost": 2.0,
            "cache_input_cost": None,  # Explicitly None
            "cache_output_cost": None,  # Explicitly None
            "context_window": 8000,
            "training_cutoff": "2023-12",
        },
    }

    with patch("llm_registry.cli.CapabilityRegistry") as mock_registry_class:
        mock_registry = mock_registry_class.return_value

        # Import the model class to create a properly structured model
        from llm_registry.models import ModelCapabilities

        mock_model = ModelCapabilities.model_validate(mock_model_data)
        mock_registry.get_model.return_value = mock_model

        # Ensure user models returns this model
        with patch("llm_registry.cli.load_user_models", return_value={"models": {"test-model": {}}}):
            result = runner.invoke(app, ["get", "test-model"])

            assert result.exit_code == 0
            # Look for specific parts that would be rendered for None cache costs
            assert "Cache Input Cost" not in result.stdout
            assert "Cache Output Cost" not in result.stdout


def test_update_command_only_model_family(runner):
    """Test update command with only model family (to hit line 395)."""
    mock_user_models = {"models": {"test-model": {"providers": ["openai"], "model_family": "original-family"}}}

    with patch("llm_registry.cli.is_package_model", return_value=False):
        with patch("llm_registry.cli.load_user_models", return_value=mock_user_models):
            with patch("llm_registry.cli.save_user_models") as mock_save:
                # Only update model family
                result = runner.invoke(app, ["update", "test-model", "--model-family", "new-family"])

                assert result.exit_code == 0
                assert "Updated model 'test-model'" in result.stdout

                # Verify only model_family was changed
                called_data = mock_save.call_args[0][0]
                assert called_data["models"]["test-model"]["model_family"] == "new-family"
                assert "token_costs" not in called_data["models"]["test-model"]


def test_update_command_create_api_params_structure(runner):
    """Test update command to create api_params structure (to hit line 401)."""
    mock_user_models = {
        "models": {
            "minimal-model": {
                "providers": ["openai"],
                # No api_params structure initially
            }
        }
    }

    with patch("llm_registry.cli.is_package_model", return_value=False):
        with patch("llm_registry.cli.load_user_models", return_value=mock_user_models):
            with patch("llm_registry.cli.save_user_models") as mock_save:
                # Set an API parameter on a model that doesn't have api_params yet
                result = runner.invoke(app, ["update", "minimal-model", "--stream"])

                assert result.exit_code == 0
                assert "Updated model 'minimal-model'" in result.stdout

                # Verify api_params structure was created
                called_data = mock_save.call_args[0][0]
                assert "api_params" in called_data["models"]["minimal-model"]
                assert called_data["models"]["minimal-model"]["api_params"]["stream"] is True


def test_update_command_create_features_structure(runner):
    """Test update command to create features structure (to hit line 403)."""
    mock_user_models = {
        "models": {
            "minimal-model": {
                "providers": ["openai"],
                # No features structure initially
            }
        }
    }

    with patch("llm_registry.cli.is_package_model", return_value=False):
        with patch("llm_registry.cli.load_user_models", return_value=mock_user_models):
            with patch("llm_registry.cli.save_user_models") as mock_save:
                # Set a feature on a model that doesn't have features yet
                result = runner.invoke(app, ["update", "minimal-model", "--vision"])

                assert result.exit_code == 0
                assert "Updated model 'minimal-model'" in result.stdout

                # Verify features structure was created
                called_data = mock_save.call_args[0][0]
                assert "features" in called_data["models"]["minimal-model"]
                assert called_data["models"]["minimal-model"]["features"]["vision"] is True
