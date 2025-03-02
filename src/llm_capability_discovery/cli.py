"""
CLI interface for LLM Capability Discovery.
"""

from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from llm_capability_discovery import CapabilityRegistry, CapabilityRepository, Provider
from llm_capability_discovery.utils import create_model_capability

app = typer.Typer(help="LLM Capability Discovery CLI")
console = Console()


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
    table.add_column("Provider", min_width=10, no_wrap=True)
    table.add_column("Model ID", min_width=20, no_wrap=True)
    table.add_column("Family", min_width=10, no_wrap=True)
    table.add_column("Input Cost", min_width=12, no_wrap=True)
    table.add_column("Output Cost", min_width=12, no_wrap=True)
    table.add_column("Context", min_width=8, no_wrap=True)
    table.add_column("Training", min_width=10, no_wrap=True)
    table.add_column("Streaming", min_width=9, justify="center", no_wrap=True)
    table.add_column("Tools", min_width=7, justify="center", no_wrap=True)
    table.add_column("Vision", min_width=7, justify="center", no_wrap=True)
    table.add_column("JSON", min_width=7, justify="center", no_wrap=True)
    table.add_column("System", min_width=7, justify="center", no_wrap=True)

    # Add rows without truncation
    for model in models:
        table.add_row(
            model.provider.value,
            model.model_id,
            model.model_family or "",
            f"${model.token_costs.input_cost}/1M" if model.token_costs else "N/A",
            f"${model.token_costs.output_cost}/1M" if model.token_costs else "N/A",
            str(model.token_costs.context_window) if model.token_costs and model.token_costs.context_window else "N/A",
            model.token_costs.training_cutoff if model.token_costs and model.token_costs.training_cutoff else "N/A",
            "✅" if model.supports_streaming else "❌",
            "✅" if model.supports_tools else "❌",
            "✅" if model.supports_vision else "❌",
            "✅" if model.supports_json_mode else "❌",
            "✅" if model.supports_system_prompt else "❌",
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
    streaming: bool = typer.Option(False, "--streaming", help="Supports streaming"),
    tools: bool = typer.Option(False, "--tools", help="Supports tools/function calling"),
    vision: bool = typer.Option(False, "--vision", help="Supports vision/image input"),
    json_mode: bool = typer.Option(False, "--json-mode", help="Supports JSON mode"),
    system_prompt: bool = typer.Option(False, "--system-prompt", help="Supports system prompt"),
    data_dir: Path | None = typer.Option(None, help="Data directory (default: ~/.llm-capability-discovery)"),
):
    """
    Add a new model.
    """
    repo = CapabilityRepository(data_dir)

    # Check if model already exists
    if repo.get_model_capabilities(provider, model_id):
        console.print(f"[red]Model {provider.value}/{model_id} already exists.[/red]")
        raise typer.Exit(code=1)

    # Create model capabilities
    capabilities = create_model_capability(
        model_id=model_id,
        provider=provider,
        model_family=model_family,
        input_cost=input_cost,
        output_cost=output_cost,
        context_window=context_window,
        training_cutoff=training_cutoff,
        supports_streaming=streaming,
        supports_tools=tools,
        supports_vision=vision,
        supports_json_mode=json_mode,
        supports_system_prompt=system_prompt,
    )

    # Save model capabilities
    file_path = repo.save_model_capabilities(capabilities)
    console.print("[green]Model added successfully[/green]")
    console.print(f"[green]Model {provider.value}/{model_id} saved to {file_path}[/green]")


@app.command()
def delete(
    model_id: str = typer.Argument(help="Model identifier (e.g., 'gpt-4')"),
    provider: Provider = typer.Option(help="Model provider"),
    force: bool = typer.Option(False, "--force", "-f", help="Force deletion without confirmation"),
    data_dir: Path | None = typer.Option(None, help="Data directory (default: ~/.llm-capability-discovery)"),
):
    """Delete a model."""
    repo = CapabilityRepository(data_dir=data_dir)

    # Confirm deletion
    if not force:
        if not typer.confirm(f"Delete model {provider.value}/{model_id}?"):
            console.print("[yellow]Operation canceled.[/yellow]")
            return

    # Delete model
    if repo.delete_model(provider, model_id):
        console.print(f"Model {provider.value}/{model_id} deleted.")
    else:
        console.print(f"[red]Failed to delete model {provider.value}/{model_id}.[/red]")
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
