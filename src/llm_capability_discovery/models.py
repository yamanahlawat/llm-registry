"""
Data models for LLM capabilities.
"""

from enum import Enum

from pydantic import BaseModel, Field


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
    OTHER = "other"


class TokenCost(BaseModel):
    """
    Token cost information for a model.
    """

    input_cost: float = Field(description="Cost per 1M input tokens")
    output_cost: float = Field(description="Cost per 1M output tokens")
    context_window: int | None = Field(default=None, description="Maximum context window in tokens")
    training_cutoff: str | None = Field(None, description="Training data cutoff date (e.g., 'Apr 2023')")


class ModelCapabilities(BaseModel):
    """
    Complete information about a model's capabilities.
    """

    # Identification
    model_id: str = Field(description="Unique identifier for the model")
    provider: Provider = Field(description="Provider name")
    model_family: str | None = Field(None, description="Model family (e.g., 'gpt-4', 'claude-3')")

    # Capabilities
    supports_streaming: bool = Field(False, description="Whether the model supports streaming responses")
    supports_tools: bool = Field(False, description="Whether the model supports tools/function calling")
    supports_vision: bool = Field(False, description="Whether the model supports processing images")
    supports_json_mode: bool = Field(False, description="Whether the model supports JSON mode output")
    supports_system_prompt: bool = Field(False, description="Whether the model supports system prompts")

    # Costs
    token_costs: TokenCost | None = Field(None, description="Token cost information for the model")

    def __str__(self) -> str:
        """
        Return a string representation of the model capabilities.
        """
        return (
            f"{self.provider.value}/{self.model_id} - "
            f"Streaming: {self.supports_streaming}, "
            f"Tools: {self.supports_tools}, "
            f"Vision: {self.supports_vision}, "
            f"System Prompt: {self.supports_system_prompt}"
        )
