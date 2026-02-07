"""
Tests for utility functions.
"""

import json
from pathlib import Path
from unittest.mock import mock_open, patch

from llm_registry.models import Modality, Provider
from llm_registry.utils import (
    create_model_capability,
    get_user_data_dir,
    get_user_models_file,
    is_package_model,
    load_package_models,
    load_user_models,
    normalize_provider_value,
    save_user_models,
)


def test_create_model_capability_minimal():
    """Test creating a model capability with minimal arguments."""
    model = create_model_capability(model_id="test-model", provider=Provider.OPENAI)
    assert model.model_id == "test-model"
    assert model.providers == [Provider.OPENAI]
    assert model.model_family is None
    assert model.api_params.stream is False
    assert model.features.tools is False
    assert model.token_costs is None


def test_create_model_capability_with_string_provider():
    """Test creating a model capability with a string provider."""
    model = create_model_capability(model_id="test-model", provider="openai")
    assert model.providers == [Provider.OPENAI]

    model = create_model_capability(model_id="test-model", provider="unknown-provider")
    assert model.providers == [Provider.OTHER]


def test_create_model_capability_full():
    """Test creating a full model capability with all arguments."""
    model = create_model_capability(
        model_id="test-model",
        provider=Provider.OPENAI,
        model_family="test-family",
        supports_streaming=True,
        supports_tools=True,
        supports_vision=True,
        supports_json_mode=True,
        supports_system_prompt=True,
        input_cost=1.0,
        output_cost=2.0,
        context_window=8000,
        training_cutoff="2023-12",
    )
    assert model.model_id == "test-model"
    assert model.providers == [Provider.OPENAI]
    assert model.model_family == "test-family"
    assert model.api_params.stream is True
    assert model.features.tools is True
    assert model.features.vision is True
    assert model.features.json_mode is True
    assert model.features.system_prompt is True
    assert model.token_costs.input_cost == 1.0
    assert model.token_costs.output_cost == 2.0
    assert model.token_costs.context_window == 8000
    assert model.token_costs.training_cutoff == "2023-12"


def test_get_user_data_dir():
    """Test getting user data directory."""
    with patch("llm_registry.utils.Path.home", return_value=Path("/mock/home")):
        with patch("llm_registry.utils.Path.mkdir") as mock_mkdir:
            data_dir = get_user_data_dir()
            assert data_dir == Path("/mock/home/.llm-registry")
            mock_mkdir.assert_called_once_with(parents=True, exist_ok=True)


def test_get_user_models_file_existing():
    """Test getting user models file when it exists."""
    with patch("llm_registry.utils.get_user_data_dir", return_value=Path("/mock/data/dir")):
        with patch("llm_registry.utils.Path.exists", return_value=True):
            models_file = get_user_models_file()
            assert models_file == Path("/mock/data/dir/models.json")


def test_get_user_models_file_not_existing():
    """Test getting user models file when it doesn't exist."""
    with patch("llm_registry.utils.get_user_data_dir", return_value=Path("/mock/data/dir")):
        with patch("llm_registry.utils.Path.exists", return_value=False):
            with patch("builtins.open", mock_open()) as mock_file:
                with patch("pathlib.Path.mkdir"):  # Ensure directory creation is mocked
                    models_file = get_user_models_file()
                    assert models_file == Path("/mock/data/dir/models.json")
                    mock_file.assert_called_once_with(Path("/mock/data/dir/models.json"), "w", encoding="utf-8")
                    # Check that some writing occurred (don't assert exact number of writes)
                    handle = mock_file()
                    assert handle.write.called
                    # Verify the content by examining all write calls
                    written_calls = [call[0][0] for call in handle.write.call_args_list]
                    written_content = "".join(written_calls)
                    assert '"models": {}' in written_content


def test_load_package_models():
    """Test loading package models."""
    mock_data = {"models": {"test-model": {}}}

    with patch("builtins.open", mock_open(read_data=json.dumps(mock_data))):
        data = load_package_models()
        assert data == mock_data


def test_load_user_models():
    """Test loading user models."""
    mock_data = {"models": {"test-model": {}}}

    with patch("llm_registry.utils.get_user_models_file", return_value=Path("/mock/data/dir/models.json")):
        with patch("builtins.open", mock_open(read_data=json.dumps(mock_data))):
            data = load_user_models()
            assert data == mock_data


def test_load_user_models_json_error():
    """Test loading user models with JSON error."""
    with patch("llm_registry.utils.get_user_models_file", return_value=Path("/mock/data/dir/models.json")):
        with patch("builtins.open", mock_open(read_data="invalid json")):
            data = load_user_models()
            assert data == {"models": {}}  # Should return default empty structure


def test_save_user_models():
    """Test saving user models."""
    mock_data = {"models": {"test-model": {}}}

    with patch("llm_registry.utils.get_user_models_file", return_value=Path("/mock/data/dir/models.json")):
        with patch("builtins.open", mock_open()) as mock_file:
            with patch("pathlib.Path.mkdir"):  # Ensure directory creation is mocked
                save_user_models(mock_data)
                mock_file.assert_called_once_with(Path("/mock/data/dir/models.json"), "w", encoding="utf-8")
                handle = mock_file()
                assert handle.write.called

                # Verify the content by examining all write calls
                written_calls = [call[0][0] for call in handle.write.call_args_list]
                written_content = "".join(written_calls)
                assert '"test-model": {}' in written_content


def test_save_user_models_no_models_key():
    """Test saving user models when models key is missing."""
    mock_data = {"other_key": "value"}

    with patch("llm_registry.utils.get_user_models_file", return_value=Path("/mock/data/dir/models.json")):
        with patch("builtins.open", mock_open()) as mock_file:
            with patch("pathlib.Path.mkdir"):  # Ensure directory creation is mocked
                save_user_models(mock_data)
                handle = mock_file()
                assert handle.write.called

                # Verify the content by examining all write calls
                written_calls = [call[0][0] for call in handle.write.call_args_list]
                written_content = "".join(written_calls)
                assert '"other_key": "value"' in written_content
                assert '"models": {}' in written_content


def test_is_package_model():
    """Test checking if a model is a package model."""
    mock_data = {"models": {"gpt-4o": {}, "claude-3": {}}}

    with patch("llm_registry.utils.load_package_models", return_value=mock_data):
        assert is_package_model("gpt-4o") is True
        assert is_package_model("claude-3") is True
        assert is_package_model("custom-model") is False


def test_normalize_provider_value():
    """Test normalizing provider values to strings."""
    assert normalize_provider_value(Provider.OPENAI) == "openai"
    assert normalize_provider_value(Provider.ANTHROPIC) == "anthropic"
    assert normalize_provider_value("openai") == "openai"
    assert normalize_provider_value("custom-provider") == "custom-provider"


def test_create_model_capability_with_provider_list():
    """Test creating a model capability with a list of providers."""
    model = create_model_capability(model_id="test-model", provider=[Provider.OPENAI, Provider.ANTHROPIC])
    assert model.providers == [Provider.OPENAI, Provider.ANTHROPIC]

    model = create_model_capability(model_id="test-model", provider=["openai", "anthropic"])
    assert model.providers == [Provider.OPENAI, Provider.ANTHROPIC]

    model = create_model_capability(model_id="test-model", provider=["openai", "unknown-provider"])
    assert model.providers == [Provider.OPENAI, Provider.OTHER]


def test_create_model_capability_with_modalities_and_pricing_dimensions():
    """Test creating a model capability with modality and pricing dimension data."""
    model = create_model_capability(
        model_id="image-model",
        provider=Provider.OPENAI,
        input_modalities=["text", "image"],
        output_modalities=["image"],
        pricing_dimensions=[
            {
                "name": "image_output_standard",
                "modality": "image",
                "direction": "output",
                "unit": "per_image",
                "price_usd": 0.04,
            }
        ],
    )
    assert model.modalities.input == [Modality.TEXT, Modality.IMAGE]
    assert model.modalities.output == [Modality.IMAGE]
    assert model.pricing_dimensions is not None
    assert model.pricing_dimensions[0].name == "image_output_standard"
    assert model.pricing_dimensions[0].unit == "per_image"


def test_create_model_capability_unknown_modality_maps_to_other():
    """Test unknown modalities are normalized to Modality.OTHER."""
    model = create_model_capability(
        model_id="unknown-modality-model",
        provider=Provider.OPENAI,
        input_modalities=["nonstandard_modality"],
    )
    assert model.modalities.input == [Modality.OTHER]


def test_create_model_capability_with_explicit_pricing_dimension_objects():
    """Test creating model capability with explicitly instantiated PricingDimension objects."""
    from llm_registry.models import PricingDimension

    pricing_obj = PricingDimension(
        name="explicit_obj_test",
        modality=Modality.IMAGE,
        direction="output",
        unit="per_image",
        price_usd=0.05,
    )

    model = create_model_capability(
        model_id="explicit-obj-model",
        provider=Provider.OPENAI,
        pricing_dimensions=[pricing_obj],
    )

    assert model.pricing_dimensions is not None
    assert len(model.pricing_dimensions) == 1
    assert model.pricing_dimensions[0].name == "explicit_obj_test"
    # Ensure it wasn't somehow broken during processing
    assert model.pricing_dimensions[0].price_usd == 0.05
