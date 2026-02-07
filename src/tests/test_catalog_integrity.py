"""
Catalog integrity checks for package model data.
"""

import io
import json
from pathlib import Path

from llm_registry.models import ModelCapabilities


def _load_package_models_from_disk() -> dict:
    # Test suite globally mocks builtins.open, so use io.open directly for package fixture reads.
    models_path = Path(__file__).resolve().parents[1] / "llm_registry" / "data" / "models.json"
    with io.open(models_path, encoding="utf-8") as models_file:
        return json.load(models_file)["models"]


def test_all_package_models_validate_against_schema():
    """Every package model entry should validate as ModelCapabilities."""
    package_models = _load_package_models_from_disk()
    assert package_models

    for model_id, model_data in package_models.items():
        model = ModelCapabilities.model_validate({**model_data, "model_id": model_id})
        assert model.model_id == model_id


def test_known_past_shutdown_and_retired_models_are_absent():
    """Known past shutdown/retired model IDs should not exist in package data."""
    package_models = _load_package_models_from_disk()

    assert "gpt-4.5-preview" not in package_models
    assert "o1-mini" not in package_models
    assert "claude-3-opus" not in package_models
    assert "claude-3-5-sonnet-latest" not in package_models
    assert "deepseek-v3.2-speciale" not in package_models
    assert "mistral-small-2503" not in package_models
    assert "mistral-large-2407" not in package_models
    assert "pixtral-12b-2409" not in package_models


def test_legacy_dotted_anthropic_ids_are_absent():
    """Legacy Anthropic IDs using dotted minor format should not be present."""
    package_models = _load_package_models_from_disk()

    assert "claude-opus-4.1" not in package_models
    assert "claude-opus-4.5" not in package_models
    assert "claude-sonnet-4.5" not in package_models
    assert "claude-haiku-4.5" not in package_models


def test_recent_model_pricing_values():
    """Key recently-refreshed model prices should match package data."""
    package_models = _load_package_models_from_disk()

    assert package_models["gpt-5.1-codex-mini"]["token_costs"]["input_cost"] == 0.25
    assert package_models["gpt-5.1-codex-mini"]["token_costs"]["output_cost"] == 2.0

    assert package_models["gpt-4o-transcribe"]["token_costs"]["output_cost"] == 10.0
    assert package_models["gpt-4o-mini-transcribe"]["token_costs"]["output_cost"] == 5.0

    assert package_models["gpt-image-1-mini"]["token_costs"]["cache_input_cost"] == 0.2

    assert package_models["deepseek-chat"]["token_costs"]["input_cost"] == 0.28
    assert package_models["deepseek-reasoner"]["token_costs"]["output_cost"] == 0.42

    assert package_models["gemini-2.5-flash-preview-09-2025"]["token_costs"]["input_cost"] == 0.3
    assert package_models["gemini-2.5-flash-lite-preview-09-2025"]["token_costs"]["output_cost"] == 0.4

    assert package_models["grok-4-0709"]["token_costs"]["output_cost"] == 15.0
    assert package_models["mistral-large-2512"]["token_costs"]["input_cost"] == 0.5
