"""
Integration tests for LLM Registry.
"""

from unittest.mock import patch

import pytest

from llm_registry import CapabilityRegistry, CapabilityRepository, create_model_capability
from llm_registry.exceptions import ModelNotFoundError
from llm_registry.models import Provider


class TestIntegration:
    """Integration tests that test multiple components together."""

    @pytest.fixture
    def mock_package_models(self):
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
    def mock_user_models(self):
        return {"models": {}}

    def test_add_and_retrieve_model(self, mock_package_models, mock_user_models):
        """Test end-to-end flow of adding a model and retrieving it."""
        # Setup mocks
        with patch("llm_registry.repository.load_user_models", return_value=mock_user_models):
            with patch("llm_registry.repository.save_user_models") as mock_save:
                with patch("llm_registry.registry.load_package_models", return_value=mock_package_models):
                    with patch("llm_registry.registry.load_user_models", return_value=mock_user_models):
                        # Create a model via the helper
                        model = create_model_capability(
                            model_id="custom-model",
                            provider=Provider.ANTHROPIC,
                            model_family="custom-family",
                            supports_streaming=True,
                            supports_tools=True,
                            input_cost=3.0,
                            output_cost=15.0,
                            context_window=200000,
                            training_cutoff="2024-02",
                        )

                        # Save it via repository
                        repo = CapabilityRepository()
                        repo.save_model_capabilities(model)

                        # Mock save_user_models will be called with updated data
                        mock_save.assert_called()

                        # Updated the mocked data to simulate the saving operation
                        updated_user_models = mock_user_models.copy()
                        updated_user_models["models"] = updated_user_models.get("models", {}).copy()
                        updated_user_models["models"]["custom-model"] = model.model_dump(exclude={"model_id"})

                        # Now retry with updated data
                        with patch("llm_registry.registry.load_user_models", return_value=updated_user_models):
                            # Now retrieve via registry
                            registry = CapabilityRegistry()
                            retrieved_model = registry.get_model("custom-model")

                            # Verify retrieved model matches original
                            assert retrieved_model.model_id == "custom-model"
                            assert retrieved_model.providers == [Provider.ANTHROPIC]
                            assert retrieved_model.model_family == "custom-family"
                            assert retrieved_model.api_params.stream is True
                            assert retrieved_model.features.tools is True
                            assert retrieved_model.token_costs.input_cost == 3.0
                            assert retrieved_model.token_costs.output_cost == 15.0
                            assert retrieved_model.token_costs.context_window == 200000
                            assert retrieved_model.token_costs.training_cutoff == "2024-02"

    def test_user_model_overrides_package_model(self, mock_package_models, mock_user_models):
        """Test that user model properly overrides package model."""
        # Add a user version of the gpt-4o model with different settings
        mock_user_models["models"]["gpt-4o"] = {
            "providers": ["openai", "azure"],  # Added new provider
            "model_family": "gpt-4-custom",  # Changed family
            "api_params": {"stream": True},
            "features": {
                "vision": False,  # Changed feature
                "tools": True,
                "json_mode": True,
                "system_prompt": True,
            },
            "token_costs": {
                "input_cost": 1.0,  # Changed cost
                "output_cost": 5.0,  # Changed cost
                "context_window": 128000,
                "training_cutoff": "2024-01",
            },
        }

        # Create the registry directly with our mock data
        registry = CapabilityRegistry()

        # Mock the internal attributes of the registry with our test data
        registry._package_models = mock_package_models
        registry._user_models = mock_user_models

        # Get model from registry
        model = registry.get_model("gpt-4o")

        # Verify user settings take precedence
        assert model.model_family == "gpt-4-custom"
        assert Provider.AZURE in model.providers
        assert model.features.vision is False
        assert model.token_costs.input_cost == 1.0
        assert model.token_costs.output_cost == 5.0

    def test_repository_and_registry_workflow(self, mock_package_models, mock_user_models):
        """Test workflow using both repository and registry."""
        # Create a model to add
        model = create_model_capability(
            model_id="test-model",
            provider=Provider.OPENAI,
            model_family="test-family",
            supports_streaming=True,
            input_cost=1.0,
            output_cost=2.0,
        )

        # Setup mocks for the repository
        with patch("llm_registry.repository.load_user_models", return_value=mock_user_models.copy()):
            with patch("llm_registry.repository.save_user_models") as mock_save:
                repo = CapabilityRepository()
                repo.save_model_capabilities(model)

                # Verify save was called
                mock_save.assert_called_once()

        # Create updated user models that include our new model
        updated_user_models = mock_user_models.copy()
        updated_user_models["models"] = updated_user_models.get("models", {}).copy()
        updated_user_models["models"]["test-model"] = model.model_dump(exclude={"model_id"})

        # Create a registry with our mocked data
        registry = CapabilityRegistry()
        registry._package_models = mock_package_models
        registry._user_models = updated_user_models

        # Test model retrieval
        test_model = registry.get_model("test-model")
        assert test_model.model_id == "test-model"
        assert test_model.model_family == "test-family"

        # Test package model retrieval
        gpt4o_model = registry.get_model("gpt-4o")
        assert gpt4o_model.model_id == "gpt-4o"

        # Test listing models
        all_models = registry.get_models()
        model_ids = [m.model_id for m in all_models]
        assert "test-model" in model_ids
        assert "gpt-4o" in model_ids

        # Test filtering models
        openai_models = registry.get_models(provider=Provider.OPENAI)
        openai_model_ids = [m.model_id for m in openai_models]
        assert "test-model" in openai_model_ids
        assert "gpt-4o" in openai_model_ids

        # Now simulate model deletion
        # First, create another copy of user models
        final_user_models = updated_user_models.copy()
        final_user_models["models"] = final_user_models["models"].copy()

        # Setup repository for deletion
        with patch("llm_registry.repository.load_user_models", return_value=final_user_models):
            with patch("llm_registry.repository.save_user_models") as mock_save:
                repo = CapabilityRepository()
                repo.delete_model(Provider.OPENAI, "test-model")
                mock_save.assert_called_once()

                # Check that save_user_models was called with data that doesn't include the deleted model
                saved_data = mock_save.call_args[0][0]
                assert "test-model" not in saved_data["models"]

        # Create a new dictionary that simulates the deletion
        post_deletion_models = {"models": {k: v for k, v in final_user_models["models"].items() if k != "test-model"}}

        # Create a new registry with the updated (post-deletion) data
        final_registry = CapabilityRegistry()
        final_registry._package_models = mock_package_models
        final_registry._user_models = post_deletion_models

        # Test that deleted model raises ModelNotFoundError
        with pytest.raises(ModelNotFoundError) as exc_info:
            final_registry.get_model("test-model")
        assert str(exc_info.value) == "Model 'test-model' not found in registry"
