"""
Data models for LLM capabilities.
"""

from enum import Enum
from typing import List, Self

from pydantic import BaseModel, Field, model_validator


class Provider(str, Enum):
    """
    Enumeration of LLM providers.
    """

    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    COHERE = "cohere"
    MISTRAL = "mistral"
    META = "meta"
    AI21 = "ai21"
    OLLAMA = "ollama"
    GITHUB = "github"
    AZURE = "azure"
    MICROSOFT = "microsoft"
    DEEPSEEK = "deepseek"
    XAI = "xai"
    AMAZON = "amazon"
    OTHER = "other"
    ALIBABA = "alibaba"


class TokenCost(BaseModel):
    """
    Token cost information for a model.
    """

    input_cost: float = Field(description="Cost per 1M input tokens")
    output_cost: float = Field(description="Cost per 1M output tokens")
    cache_input_cost: float | None = Field(default=None, description="Cost per 1M cached input tokens")
    cache_output_cost: float | None = Field(default=None, description="Cost per 1M cached output tokens")
    context_window: int | None = Field(default=None, description="Maximum context window in tokens")
    training_cutoff: str | None = Field(default=None, description="Training data cutoff date (e.g., 'Apr 2023')")

    @model_validator(mode="after")
    def check_passwords_match(self) -> Self:
        if self.cache_input_cost and self.cache_input_cost > self.input_cost:
            raise ValueError("Cache input cost must be lower than input cost")
        if self.cache_output_cost and self.cache_output_cost > self.output_cost:
            raise ValueError("Cache output cost must be lower than output cost")
        return self


class ApiParams(BaseModel):
    """
    API parameters supported by the model.
    """

    max_tokens: bool = Field(default=False, description="Whether the model supports max_tokens parameter")
    temperature: bool = Field(default=False, description="Whether the model supports temperature parameter")
    top_p: bool = Field(default=False, description="Whether the model supports top_p parameter")
    stream: bool = Field(default=False, description="Whether the model supports streaming responses")


class Features(BaseModel):
    """
    High-level features supported by the model.
    """

    vision: bool = Field(default=False, description="Whether the model supports processing images")
    tools: bool = Field(default=False, description="Whether the model supports tools/function calling")
    json_mode: bool = Field(default=False, description="Whether the model supports JSON mode output")
    system_prompt: bool = Field(default=False, description="Whether the model supports system prompts")


class ModelCapabilities(BaseModel):
    """
    Complete information about a model's capabilities.
    """

    model_id: str = Field(description="Unique identifier for the model")
    providers: List[Provider] = Field(description="List of providers that support this model")
    model_family: str | None = Field(default=None, description="Model family (e.g., 'gpt-4', 'claude-3')")
    base_model: str | None = Field(default=None, description="For open source models, the original model name")
    api_params: ApiParams = Field(default_factory=ApiParams, description="API parameters supported by the model")
    features: Features = Field(
        default_factory=lambda: Features(), description="High-level features supported by the model"
    )
    token_costs: TokenCost | None = Field(default=None, description="Token cost information for the model")

    def __str__(self) -> str:
        """
        Return a string representation of the model capabilities.
        """
        providers_str = ", ".join(p.value for p in self.providers)
        return (
            f"Providers: {providers_str} - "
            f"Family: {self.model_family} - "
            f"Features: Vision={self.features.vision}, "
            f"Tools={self.features.tools}, "
            f"JSON={self.features.json_mode}, "
            f"System={self.features.system_prompt}"
        )
