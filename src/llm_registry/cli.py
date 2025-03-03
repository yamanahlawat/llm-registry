"""
CLI interface for LLM Registry.
"""

import json
from functools import lru_cache
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from llm_registry import CapabilityRegistry, Provider
from llm_registry.models import ApiParams, Features, ModelCapabilities, TokenCost

app = typer.Typer(help="LLM Registry CLI")
console = Console()


@lru_cache()
def get_models_file() -> Path:
    """Get the path to models.json file."""
    return Path(__file__).parent / "data" / "models.json"


@lru_cache()
def load_models() -> dict:
    """Load models from JSON file with caching."""
    with open(get_models_file()) as f:
        return json.load(f)


def save_models(data: dict) -> None:
    """Save models to JSON file and invalidate cache."""
    models_file = get_models_file()
    with open(models_file, "w") as f:
        json.dump(data, f, indent=2)
    # Invalidate cache
    load_models.cache_clear()


def get_model_data(model_id: str, data: dict | None = None) -> tuple[dict, dict]:
    """Get model data and validate it exists."""
    if data is None:
        data = load_models()

    if model_id not in data["models"]:
        console.print(f"[red]Error: Model '{model_id}' not found in registry.[/red]")
        raise typer.Exit(code=1)

    return data, data["models"][model_id]


def validate_provider(model_data: dict, provider: Provider, model_id: str) -> None:
    """Validate provider exists for model."""
    if provider and provider.value not in model_data["providers"]:
        console.print(f"[red]Error: Provider '{provider.value}' not found for model '{model_id}'.[/red]")
        raise typer.Exit(code=1)


@app.command()
def list(
    provider: Provider | None = typer.Option(None, help="Filter by provider"),
):
    """
    List available models.
    """
    registry = CapabilityRegistry(offline_mode=True)
    models = registry.get_models(provider=provider)

    if not models:
        console.print("No models found")
        return

    # Create table with minimal formatting
    table = Table(title="LLM Model Capabilities", show_lines=True, header_style="bold", show_header=True, expand=False)
    table.add_column("Model ID", min_width=20, no_wrap=True)
    table.add_column("Providers", min_width=15, no_wrap=True)
    table.add_column("Family", min_width=10, no_wrap=True)
    table.add_column("Input $/M", min_width=12, no_wrap=True)
    table.add_column("Output $/M", min_width=12, no_wrap=True)
    table.add_column("Cache In $/M", min_width=12, no_wrap=True)
    table.add_column("Cache Out $/M", min_width=12, no_wrap=True)
    table.add_column("Context", min_width=8, no_wrap=True)
    table.add_column("Training", min_width=10, no_wrap=True)
    table.add_column("Streaming", min_width=9, justify="center", no_wrap=True)
    table.add_column("Tools", min_width=7, justify="center", no_wrap=True)
    table.add_column("Vision", min_width=7, justify="center", no_wrap=True)
    table.add_column("JSON", min_width=7, justify="center", no_wrap=True)
    table.add_column("System", min_width=7, justify="center", no_wrap=True)

    # Sort models by ID
    models.sort(key=lambda x: x.model_id)

    for model in models:
        providers_str = ", ".join(p.value for p in model.providers)
        table.add_row(
            model.model_id,
            providers_str,
            model.model_family or "",
            f"${model.token_costs.input_cost}" if model.token_costs else "N/A",
            f"${model.token_costs.output_cost}" if model.token_costs else "N/A",
            f"${model.token_costs.cache_input_cost}"
            if model.token_costs and model.token_costs.cache_input_cost is not None
            else "N/A",
            f"${model.token_costs.cache_output_cost}"
            if model.token_costs and model.token_costs.cache_output_cost is not None
            else "N/A",
            str(model.token_costs.context_window) if model.token_costs and model.token_costs.context_window else "N/A",
            model.token_costs.training_cutoff if model.token_costs and model.token_costs.training_cutoff else "N/A",
            "✅" if model.api_params.stream else "❌",
            "✅" if model.features.tools else "❌",
            "✅" if model.features.vision else "❌",
            "✅" if model.features.json_mode else "❌",
            "✅" if model.features.system_prompt else "❌",
        )

    # Print table with full width
    console.print(table, width=200)


@app.command()
def add(
    model_id: str = typer.Argument(..., help="Model identifier (e.g., 'gpt-4')"),
    provider: Provider = typer.Option(..., help="Model provider"),
    model_family: str | None = typer.Option(None, help="Model family (e.g., 'GPT-4')"),
    input_cost: float | None = typer.Option(None, help="Input token cost per 1M tokens"),
    output_cost: float | None = typer.Option(None, help="Output token cost per 1M tokens"),
    context_window: int | None = typer.Option(None, help="Context window size in tokens"),
    training_cutoff: str | None = typer.Option(None, help="Training data cutoff date (e.g., '2024-01')"),
    # API Parameters
    max_tokens: bool = typer.Option(False, "--max-tokens", help="Supports max_tokens parameter"),
    temperature: bool = typer.Option(False, "--temperature", help="Supports temperature parameter"),
    top_p: bool = typer.Option(False, "--top-p", help="Supports top_p parameter"),
    frequency_penalty: bool = typer.Option(False, "--frequency-penalty", help="Supports frequency_penalty parameter"),
    presence_penalty: bool = typer.Option(False, "--presence-penalty", help="Supports presence_penalty parameter"),
    stop: bool = typer.Option(False, "--stop", help="Supports stop parameter"),
    n: bool = typer.Option(False, "--n", help="Supports generating multiple completions"),
    stream: bool = typer.Option(False, "--stream", help="Supports streaming"),
    # Features
    tools: bool = typer.Option(False, "--tools", help="Supports tools/function calling"),
    vision: bool = typer.Option(False, "--vision", help="Supports vision/image input"),
    json_mode: bool = typer.Option(False, "--json-mode", help="Supports JSON mode"),
    system_prompt: bool = typer.Option(False, "--system-prompt", help="Supports system prompt"),
):
    """Add a new model."""
    # Create API parameters
    api_params = ApiParams(
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        frequency_penalty=frequency_penalty,
        presence_penalty=presence_penalty,
        stop=stop,
        n=n,
        stream=stream,
    )

    # Create features
    features = Features(
        tools=tools,
        vision=vision,
        json_mode=json_mode,
        system_prompt=system_prompt,
    )

    # Create token costs if any cost parameters provided
    token_costs = None
    if any(x is not None for x in [input_cost, output_cost, context_window, training_cutoff]):
        token_costs = TokenCost(
            input_cost=input_cost or 0.0,
            output_cost=output_cost or 0.0,
            context_window=context_window,
            training_cutoff=training_cutoff,
        )

    # Create model capabilities
    capabilities = ModelCapabilities(
        model_id=model_id,
        providers=[provider],
        model_family=model_family,
        api_params=api_params,
        features=features,
        token_costs=token_costs,
    )

    # Load current data
    data = load_models()

    # Check if model already exists
    for model in data["models"].values():
        if model["model_id"] == model_id:
            if provider.value in model["providers"]:
                console.print(f"[red]Error: Model '{provider.value}/{model_id}' already exists.[/red]")
                raise typer.Exit(code=1)
            else:
                # Add provider to existing model
                model["providers"].append(provider.value)
                console.print(f"[yellow]Added provider '{provider.value}' to existing model '{model_id}'.[/yellow]")
                save_models(data)
                return

    # Add new model to models.json
    data["models"][model_id] = capabilities.model_dump()
    save_models(data)
    console.print(f"[green]Added new model '{model_id}' with provider '{provider.value}'.[/green]")


@app.command()
def delete(
    model_id: str = typer.Argument(help="Model identifier (e.g., 'gpt-4')"),
    provider: Provider | None = typer.Option(None, help="Provider to remove (if None, deletes entire model)"),
    force: bool = typer.Option(False, "--force", "-f", help="Force deletion without confirmation"),
):
    """Delete a model or remove a provider from a model."""
    # Load current data and validate model exists
    data, model_data = get_model_data(model_id)

    # If provider specified, only remove that provider
    if provider:
        validate_provider(model_data, provider, model_id)

        # Confirm deletion of provider
        if not force and not typer.confirm(f"Remove provider '{provider.value}' from model '{model_id}'?"):
            console.print("[yellow]Operation canceled.[/yellow]")
            return

        # Remove provider
        model_data["providers"].remove(provider.value)

        # If no providers left, delete the model
        if not model_data["providers"]:
            del data["models"][model_id]
            console.print(f"[green]Model '{model_id}' deleted (no providers remaining).[/green]")
        else:
            console.print(f"[green]Removed provider '{provider.value}' from model '{model_id}'.[/green]")
    else:
        # Confirm deletion of entire model
        if not force and not typer.confirm(f"Delete model '{model_id}'?"):
            console.print("[yellow]Operation canceled.[/yellow]")
            return

        # Delete model
        del data["models"][model_id]
        console.print(f"[green]Model '{model_id}' deleted.[/green]")

    # Save updated data
    save_models(data)


@app.command()
def update(
    model_id: str = typer.Argument(..., help="Model identifier (e.g., 'gpt-4')"),
    provider: Provider | None = typer.Option(None, help="Update specific provider"),
    model_family: str | None = typer.Option(None, help="Model family (e.g., 'GPT-4')"),
    input_cost: float | None = typer.Option(None, help="Input token cost per 1M tokens"),
    output_cost: float | None = typer.Option(None, help="Output token cost per 1M tokens"),
    cache_input_cost: float | None = typer.Option(None, help="Cached input token cost per 1M tokens"),
    cache_output_cost: float | None = typer.Option(None, help="Cached output token cost per 1M tokens"),
    context_window: int | None = typer.Option(None, help="Context window size in tokens"),
    training_cutoff: str | None = typer.Option(None, help="Training data cutoff date (e.g., '2024-01')"),
    # API Parameters
    max_tokens: bool | None = typer.Option(None, "--max-tokens", help="Supports max_tokens parameter"),
    temperature: bool | None = typer.Option(None, "--temperature", help="Supports temperature parameter"),
    top_p: bool | None = typer.Option(None, "--top-p", help="Supports top_p parameter"),
    frequency_penalty: bool | None = typer.Option(
        None, "--frequency-penalty", help="Supports frequency_penalty parameter"
    ),
    presence_penalty: bool | None = typer.Option(
        None, "--presence-penalty", help="Supports presence_penalty parameter"
    ),
    stop: bool | None = typer.Option(None, "--stop", help="Supports stop parameter"),
    n: bool | None = typer.Option(None, "--n", help="Supports generating multiple completions"),
    stream: bool | None = typer.Option(None, "--stream", help="Supports streaming"),
    # Features
    tools: bool | None = typer.Option(None, "--tools", help="Supports tools/function calling"),
    vision: bool | None = typer.Option(None, "--vision", help="Supports vision/image input"),
    json_mode: bool | None = typer.Option(None, "--json-mode", help="Supports JSON mode"),
    system_prompt: bool | None = typer.Option(None, "--system-prompt", help="Supports system prompt"),
):
    """Update a model's capabilities."""
    # Load current data and validate model exists
    data, model_data = get_model_data(model_id)

    # Update provider-specific fields
    validate_provider(model_data, provider, model_id)

    # Update fields if provided
    if model_family is not None:
        model_data["model_family"] = model_family

    # Update token costs
    if any(
        x is not None
        for x in [input_cost, output_cost, cache_input_cost, cache_output_cost, context_window, training_cutoff]
    ):
        if "token_costs" not in model_data:
            model_data["token_costs"] = {}
        if input_cost is not None:
            model_data["token_costs"]["input_cost"] = input_cost
        if output_cost is not None:
            model_data["token_costs"]["output_cost"] = output_cost
        if cache_input_cost is not None:
            model_data["token_costs"]["cache_input_cost"] = cache_input_cost
        if cache_output_cost is not None:
            model_data["token_costs"]["cache_output_cost"] = cache_output_cost
        if context_window is not None:
            model_data["token_costs"]["context_window"] = context_window
        if training_cutoff is not None:
            model_data["token_costs"]["training_cutoff"] = training_cutoff

    # Update API parameters
    api_params = [max_tokens, temperature, top_p, frequency_penalty, presence_penalty, stop, n, stream]
    if any(x is not None for x in api_params):
        if "api_params" not in model_data:
            model_data["api_params"] = {}
        if max_tokens is not None:
            model_data["api_params"]["max_tokens"] = max_tokens
        if temperature is not None:
            model_data["api_params"]["temperature"] = temperature
        if top_p is not None:
            model_data["api_params"]["top_p"] = top_p
        if frequency_penalty is not None:
            model_data["api_params"]["frequency_penalty"] = frequency_penalty
        if presence_penalty is not None:
            model_data["api_params"]["presence_penalty"] = presence_penalty
        if stop is not None:
            model_data["api_params"]["stop"] = stop
        if n is not None:
            model_data["api_params"]["n"] = n
        if stream is not None:
            model_data["api_params"]["stream"] = stream

    # Update features
    features = [tools, vision, json_mode, system_prompt]
    if any(x is not None for x in features):
        if "features" not in model_data:
            model_data["features"] = {}
        if tools is not None:
            model_data["features"]["tools"] = tools
        if vision is not None:
            model_data["features"]["vision"] = vision
        if json_mode is not None:
            model_data["features"]["json_mode"] = json_mode
        if system_prompt is not None:
            model_data["features"]["system_prompt"] = system_prompt

    # Save updated data
    save_models(data)
    console.print(f"[green]Updated model '{model_id}'.[/green]")


if __name__ == "__main__":
    app()
