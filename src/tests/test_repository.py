"""Tests for the CapabilityRepository class."""

import pytest

from llm_registry import CapabilityRepository, Provider
from llm_registry.utils import create_model_capability


@pytest.fixture
def temp_repo(tmp_path):
    """Create a temporary repository."""
    return CapabilityRepository(tmp_path)


@pytest.fixture
def sample_model():
    """Create a sample model capability."""
    return create_model_capability(
        model_id="test-model",
        provider=Provider.OPENAI,  # This will be converted to providers list internally
        model_family="test",
        supports_streaming=True,
        supports_tools=True,
        input_cost=0.1,
        output_cost=0.2,
        context_window=1000,
    )


def test_save_and_get_model(temp_repo, sample_model):
    """Test saving and retrieving a model."""
    # Save model
    file_path = temp_repo.save_model_capabilities(sample_model)
    assert file_path.exists()

    # Get model back
    retrieved = temp_repo.get_model_capabilities(sample_model.providers[0], sample_model.model_id)
    assert retrieved is not None
    assert retrieved.model_id == sample_model.model_id
    assert Provider.OPENAI in retrieved.providers
    assert retrieved.token_costs.input_cost == sample_model.token_costs.input_cost


def test_list_models(temp_repo, sample_model):
    """Test listing models."""
    # Save model
    temp_repo.save_model_capabilities(sample_model)

    # List all models
    models = temp_repo.list_models()
    assert len(models) == 1
    assert models[0].model_id == sample_model.model_id

    # List by provider
    models = temp_repo.list_models(provider=sample_model.providers[0])
    assert len(models) == 1
    models = temp_repo.list_models(provider=Provider.ANTHROPIC)
    assert len(models) == 0


def test_delete_model(temp_repo, sample_model):
    """Test deleting a model."""
    # Save model
    temp_repo.save_model_capabilities(sample_model)

    # Delete model
    success = temp_repo.delete_model(sample_model.providers[0], sample_model.model_id)
    assert success is True

    # Verify deletion
    model = temp_repo.get_model_capabilities(sample_model.providers[0], sample_model.model_id)
    assert model is None

    # Try deleting non-existent model
    success = temp_repo.delete_model(Provider.ANTHROPIC, "non-existent")
    assert success is False


def test_overwrite_model(temp_repo, sample_model):
    """Test overwriting an existing model."""
    # Save model
    temp_repo.save_model_capabilities(sample_model)

    # Modify and save again
    modified_model = create_model_capability(
        model_id=sample_model.model_id,
        provider=sample_model.providers[0],  # Use the same provider
        input_cost=0.2,  # Changed cost
        output_cost=0.4,
    )
    temp_repo.save_model_capabilities(modified_model)

    # Verify changes
    retrieved = temp_repo.get_model_capabilities(modified_model.providers[0], modified_model.model_id)
    assert retrieved.token_costs.input_cost == 0.2
    assert retrieved.token_costs.output_cost == 0.4
