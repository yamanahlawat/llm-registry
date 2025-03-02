"""
Utility functions for the LLM capability discovery package.
"""

from typing import Optional, Union

from .models import ModelCapabilities, Provider, TokenCost


def create_model_capability(
    model_id: str,
    provider: Union[Provider, str],
    model_family: Optional[str] = None,
    supports_streaming: bool = False,
    supports_tools: bool = False,
    supports_vision: bool = False,
    supports_json_mode: bool = False,
    supports_system_prompt: bool = False,
    input_cost: Optional[float] = None,
    output_cost: Optional[float] = None,
    context_window: Optional[int] = None,
    training_cutoff: Optional[str] = None,
) -> ModelCapabilities:
    """
    Helper function to create a ModelCapabilities object with less verbose syntax.

    Args:
        model_id: Model identifier
        provider: Model provider
        model_family: Optional model family
        supports_streaming: Whether the model supports streaming
        supports_tools: Whether the model supports tools/function calling
        supports_vision: Whether the model supports vision inputs
        supports_json_mode: Whether the model supports JSON mode
        supports_system_prompt: Whether the model supports system prompts
        input_cost: Cost per 1M input tokens
        output_cost: Cost per 1M output tokens
        context_window: Context window size in tokens
        training_cutoff: Training data cutoff date

    Returns:
        ModelCapabilities object
    """
    # Convert string provider to enum if needed
    if isinstance(provider, str):
        try:
            provider = Provider(provider)
        except ValueError:
            provider = Provider.OTHER

    # Create token costs if applicable
    token_costs = None
    if input_cost is not None and output_cost is not None:
        token_costs = TokenCost(
            input_cost=input_cost,
            output_cost=output_cost,
            context_window=context_window,
            training_cutoff=training_cutoff,
        )

    return ModelCapabilities(
        model_id=model_id,
        provider=provider,
        model_family=model_family,
        supports_streaming=supports_streaming,
        supports_tools=supports_tools,
        supports_vision=supports_vision,
        supports_json_mode=supports_json_mode,
        supports_system_prompt=supports_system_prompt,
        token_costs=token_costs,
    )
