"""
LLM Registry - A tool for discovering, testing, and sharing LLM model capabilities.
"""

from .exceptions import ModelNotFoundError
from .models import Modality, ModelCapabilities, ModelModalities, PricingDimension, Provider, TokenCost
from .registry import CapabilityRegistry
from .repository import CapabilityRepository
from .utils import create_model_capability

__version__ = "0.5.0"

__all__ = [
    "ModelCapabilities",
    "ModelModalities",
    "PricingDimension",
    "Modality",
    "Provider",
    "TokenCost",
    "CapabilityRegistry",
    "CapabilityRepository",
    "create_model_capability",
    "ModelNotFoundError",
]
