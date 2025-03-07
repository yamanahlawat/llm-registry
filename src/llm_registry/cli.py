"""
CLI interface for LLM Registry.
"""

import typer
from rich.console import Console
from rich.table import Table

from llm_registry import CapabilityRegistry, Provider
from llm_registry.exceptions import ModelNotFoundError
from llm_registry.models import ApiParams, Features, ModelCapabilities, TokenCost
from llm_registry.utils import is_package_model, load_package_models, load_user_models, save_user_models

app = typer.Typer(help="LLM Registry CLI")
console = Console()


def get_model_data(model_id: str, data: dict | None = None) -> tuple[dict, dict]:
    """
    Get model data and validate it exists.
    """
    if data is None:
        data = load_user_models()

    if model_id not in data["models"]:
        console.print(f"[red]Error: Model '{model_id}' not found in user registry.[/red]")
        raise typer.Exit(code=1)

    return data, data["models"][model_id]


def validate_provider(model_data: dict, provider: Provider, model_id: str) -> None:
    """
    Validate provider exists for model.
    """
    if provider and provider.value not in model_data["providers"]:
        console.print(f"[red]Error: Provider '{provider.value}' not found for model '{model_id}'.[/red]")
        raise typer.Exit(code=1)


def validate_not_package_model(model_id: str, action: str = "modify") -> None:
    """
    Validate that a model is not a package model.
    """
    if is_package_model(model_id):
        console.print(
            f"[red]Error: Cannot {action} model '{model_id}'. Package models are read-only. "
            "Create a custom model instead.[/red]"
        )
        raise typer.Exit(code=1)


@app.command()
def list(
    provider: Provider | None = typer.Option(default=None, help="Filter by provider"),
    user_only: bool = typer.Option(False, "--user-only", help="Only show user-defined models"),
    package_only: bool = typer.Option(False, "--package-only", help="Only show package models"),
) -> None:
    """
    List available models.
    """
    if user_only and package_only:
        console.print("[red]Error: Cannot specify both --user-only and --package-only[/red]")
        raise typer.Exit(code=1)

    # Get models based on flags
    if user_only:
        data = load_user_models()
        models = [
            ModelCapabilities.model_validate({**model_data, "model_id": model_id})
            for model_id, model_data in data["models"].items()
            if not provider or provider.value in model_data["providers"]
        ]
    elif package_only:
        data = load_package_models()
        models = [
            ModelCapabilities.model_validate({**model_data, "model_id": model_id})
            for model_id, model_data in data["models"].items()
            if not provider or provider.value in model_data["providers"]
        ]
    else:
        # Get all models from registry which handles both sources
        registry = CapabilityRegistry()
        models = registry.get_models(provider=provider)

    if not models:
        console.print("No models found")
        return

    # Create table with minimal formatting
    table = Table(title="LLM Model Capabilities", show_lines=True, header_style="bold", show_header=True, expand=False)
    table.add_column("Source", min_width=8, no_wrap=True)
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

    # Get user models for source check
    user_models = load_user_models()["models"]

    for model in models:
        # Check if model is from user or package
        source = "User" if model.model_id in user_models else "Package"
        providers_str = ", ".join(p.value for p in model.providers)
        table.add_row(
            source,
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
def get(
    model_id: str = typer.Argument(..., help="Model identifier (e.g., 'gpt-4')"),
    json_output: bool = typer.Option(False, "--json", help="Output in JSON format"),
) -> None:
    """Get detailed information about a model."""
    registry = CapabilityRegistry()
    try:
        model = registry.get_model(model_id)
    except ModelNotFoundError as e:
        console.print(f"[red]Error: {str(e)}[/red]")
        raise typer.Exit(code=1) from e

    if json_output:
        console.print_json(model.model_dump_json())
        return

    # Create tables for different sections of model information
    # Main Info Table
    main_table = Table(title=f"Model: {model_id}", show_header=True, header_style="bold")
    main_table.add_column("Property", style="cyan")
    main_table.add_column("Value")

    # Add source information
    source = "User" if model_id in load_user_models()["models"] else "Package"
    main_table.add_row("Source", source)
    main_table.add_row("Model Family", model.model_family or "N/A")
    main_table.add_row("Providers", ", ".join(p.value for p in model.providers))

    # Token Costs Table
    if model.token_costs:
        cost_table = Table(title="Token Costs", show_header=True, header_style="bold")
        cost_table.add_column("Metric", style="cyan")
        cost_table.add_column("Value")

        cost_table.add_row("Input Cost ($/M)", f"${model.token_costs.input_cost}")
        cost_table.add_row("Output Cost ($/M)", f"${model.token_costs.output_cost}")
        if model.token_costs.cache_input_cost is not None:
            cost_table.add_row("Cache Input Cost ($/M)", f"${model.token_costs.cache_input_cost}")
        if model.token_costs.cache_output_cost is not None:
            cost_table.add_row("Cache Output Cost ($/M)", f"${model.token_costs.cache_output_cost}")
        cost_table.add_row("Context Window", str(model.token_costs.context_window or "N/A"))
        cost_table.add_row("Training Cutoff", model.token_costs.training_cutoff or "N/A")

    # API Parameters Table
    api_table = Table(title="API Parameters", show_header=True, header_style="bold")
    api_table.add_column("Parameter", style="cyan")
    api_table.add_column("Value")

    api_table.add_row("Max Tokens", "✅" if model.api_params.max_tokens else "❌")
    api_table.add_row("Temperature", "✅" if model.api_params.temperature else "❌")
    api_table.add_row("Top P", "✅" if model.api_params.top_p else "❌")
    api_table.add_row("Streaming Support", "✅" if model.api_params.stream else "❌")

    # Features Table
    features_table = Table(title="Features", show_header=True, header_style="bold")
    features_table.add_column("Feature", style="cyan")
    features_table.add_column("Supported")

    features_table.add_row("Tools/Functions", "✅" if model.features.tools else "❌")
    features_table.add_row("Vision", "✅" if model.features.vision else "❌")
    features_table.add_row("JSON Mode", "✅" if model.features.json_mode else "❌")
    features_table.add_row("System Prompt", "✅" if model.features.system_prompt else "❌")

    # Print all tables with some spacing
    console.print(main_table)
    console.print()
    if model.token_costs:
        console.print(cost_table)
        console.print()
    console.print(api_table)
    console.print()
    console.print(features_table)


@app.command()
def add(
    model_id: str = typer.Argument(..., help="Model identifier (e.g., 'gpt-4')"),
    provider: Provider = typer.Option(..., help="Model provider"),
    model_family: str | None = typer.Option(None, help="Model family (e.g., 'GPT-4')"),
    input_cost: float | None = typer.Option(None, help="Input token cost per 1M tokens"),
    output_cost: float | None = typer.Option(None, help="Output token cost per 1M tokens"),
    cache_input_cost: float | None = typer.Option(None, help="Cached input token cost per 1M tokens"),
    cache_output_cost: float | None = typer.Option(None, help="Cached output token cost per 1M tokens"),
    context_window: int | None = typer.Option(None, help="Context window size in tokens"),
    training_cutoff: str | None = typer.Option(None, help="Training data cutoff date (e.g., '2024-01')"),
    # API Parameters
    max_tokens: bool = typer.Option(False, "--max-tokens", help="Supports max_tokens parameter"),
    temperature: bool = typer.Option(False, "--temperature", help="Supports temperature parameter"),
    top_p: bool = typer.Option(False, "--top-p", help="Supports top_p parameter"),
    stream: bool = typer.Option(False, "--stream", help="Supports streaming"),
    # Features
    tools: bool = typer.Option(False, "--tools", help="Supports tools/function calling"),
    vision: bool = typer.Option(False, "--vision", help="Supports vision/image input"),
    json_mode: bool = typer.Option(False, "--json-mode", help="Supports JSON mode"),
    system_prompt: bool = typer.Option(False, "--system-prompt", help="Supports system prompt"),
) -> None:
    """Add a new model to user's registry."""
    # Validate not modifying package model
    validate_not_package_model(model_id, "add")

    # Create API parameters
    api_params = ApiParams(
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
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
    if any(
        x is not None
        for x in [input_cost, output_cost, cache_input_cost, cache_output_cost, context_window, training_cutoff]
    ):
        token_costs = TokenCost(
            input_cost=input_cost or 0.0,
            output_cost=output_cost or 0.0,
            cache_input_cost=cache_input_cost,
            cache_output_cost=cache_output_cost,
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

    # Load user data
    data = load_user_models()

    # Check if model already exists in user data
    if model_id in data["models"]:
        model_data = data["models"][model_id]
        if provider.value in model_data["providers"]:
            console.print(f"[red]Error: Model '{provider.value}/{model_id}' already exists in user registry.[/red]")
            raise typer.Exit(code=1)
        else:
            # Add provider to existing model
            model_data["providers"].append(provider.value)
            console.print(f"[yellow]Added provider '{provider.value}' to existing model '{model_id}'.[/yellow]")
            save_user_models(data)
            return

    # Add new model to user registry
    data["models"][model_id] = capabilities.model_dump(exclude={"model_id"})
    save_user_models(data)
    console.print(f"[green]Added model '{provider.value}/{model_id}' to user registry.[/green]")


@app.command()
def delete(
    model_id: str = typer.Argument(..., help="Model identifier (e.g., 'gpt-4')"),
    provider: Provider | None = typer.Option(None, help="Provider to remove (if None, deletes entire model)"),
    force: bool = typer.Option(False, "--force", "-f", help="Force deletion without confirmation"),
) -> None:
    """Delete a model or remove a provider from user's registry."""
    # Validate not modifying package model
    validate_not_package_model(model_id, "delete")

    data, model_data = get_model_data(model_id)

    if provider:
        validate_provider(model_data, provider, model_id)

        if not force:
            confirm = typer.confirm(f"Remove provider '{provider.value}' from model '{model_id}' in user registry?")
            if not confirm:
                raise typer.Exit()

        model_data["providers"].remove(provider.value)
        if not model_data["providers"]:
            # No providers left, delete model
            del data["models"][model_id]
            console.print(f"[yellow]Deleted model '{model_id}' from user registry as it has no providers.[/yellow]")
        else:
            console.print(
                f"[green]Removed provider '{provider.value}' from model '{model_id}' in user registry.[/green]"
            )
    else:
        if not force:
            confirm = typer.confirm(f"Delete model '{model_id}' from user registry?")
            if not confirm:
                raise typer.Exit()

        del data["models"][model_id]
        console.print(f"[green]Deleted model '{model_id}' from user registry.[/green]")

    save_user_models(data)


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
    stream: bool | None = typer.Option(None, "--stream", help="Supports streaming"),
    # Features
    tools: bool | None = typer.Option(None, "--tools", help="Supports tools/function calling"),
    vision: bool | None = typer.Option(None, "--vision", help="Supports vision/image input"),
    json_mode: bool | None = typer.Option(None, "--json-mode", help="Supports JSON mode"),
    system_prompt: bool | None = typer.Option(None, "--system-prompt", help="Supports system prompt"),
):
    """Update a model's capabilities in user's registry."""
    # Validate not modifying package model
    validate_not_package_model(model_id, "update")

    data, model_data = get_model_data(model_id)

    if provider:
        validate_provider(model_data, provider, model_id)

    # Update model family if provided
    if model_family is not None:
        model_data["model_family"] = model_family

    # Update token costs if any provided
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

    # Update API parameters if any provided
    if any(x is not None for x in [max_tokens, temperature, top_p, stream]):
        if "api_params" not in model_data:
            model_data["api_params"] = {}
        if max_tokens is not None:
            model_data["api_params"]["max_tokens"] = max_tokens
        if temperature is not None:
            model_data["api_params"]["temperature"] = temperature
        if top_p is not None:
            model_data["api_params"]["top_p"] = top_p
        if stream is not None:
            model_data["api_params"]["stream"] = stream

    # Update features if any provided
    if any(x is not None for x in [tools, vision, json_mode, system_prompt]):
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

    save_user_models(data)
    console.print(f"[green]Updated model '{model_id}'.[/green]")


if __name__ == "__main__":  # pragma: no cover
    app()
