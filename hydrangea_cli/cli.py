"""CLI main entry point"""

import typer
import click
from typing import Optional
from .core.metadata import MetadataManager

app = typer.Typer(help="Hydrangea CLI - Query Hydrangea dataset")
metadata_manager = MetadataManager()


def _echo_kv(label: str, value: str) -> None:
    """Pretty print key-value without changing textual content."""
    # Keep exact textual form: "label: value"
    click.secho(f"{label}", bold=True, fg="bright_white", nl=False)
    click.secho(": ", bold=False, fg="bright_black", nl=False)
    # Value printed as-is to avoid content change
    click.secho(f"{value}", fg="bright_cyan")


def _echo_kv_aligned(pairs: list[tuple[str, str]]) -> None:
    """Print multiple key-values with aligned colon positions.
    Only adds padding spaces after labels; label/value text remains unchanged.
    """
    if not pairs:
        return
    width = max(len(k) for k, _ in pairs)
    for k, v in pairs:
        label_padded = f"{k.ljust(width)}"
        click.secho(label_padded, bold=True, fg="bright_white", nl=False)
        click.secho(": ", fg="bright_black", nl=False)
        click.secho(f"{v}", fg="bright_cyan")


@app.command()
def apps(
    classification: Optional[str] = typer.Option(None, "--classification", help="Filter applications by classification"),
    llm: Optional[str] = typer.Option(None, "--llm", help="Filter applications by LLM"),
    llm_deployment: Optional[str] = typer.Option(None, "--llm-deployment", help="Filter applications by LLM deployment environment"),
    vdb: Optional[str] = typer.Option(None, "--vdb", help="Filter applications by vector database"),
    vdb_deployment: Optional[str] = typer.Option(None, "--vdb-deployment", help="Filter applications by vector database deployment environment"),
    langchain: Optional[str] = typer.Option(None, "--langchain", help="Filter applications by LangChain"),
    language: Optional[str] = typer.Option(None, "--language", help="Filter applications by programming language")
):
    """List all applications"""
    apps_list = metadata_manager.get_apps_by_filters(
        classification=classification,
        llm=llm,
        llm_deployment=llm_deployment,
        vdb=vdb,
        vdb_deployment=vdb_deployment,
        langchain=langchain,
        language=language
    )
    
    if not apps_list:
        click.secho("No applications found.", fg="yellow")
        return
    
    for app_name in apps_list:
        click.secho(f"{app_name}", fg="bright_cyan")


@app.command()
def bids(
    app: Optional[str] = typer.Option(None, "--app", help="Filter defect IDs by application")
):
    """List all defect IDs"""
    defect_ids = metadata_manager.get_defect_ids(app=app)
    
    if not defect_ids:
        click.secho("No defect IDs found.", fg="yellow")
        return
    
    for defect_id in defect_ids:
        click.secho(f"{defect_id}", fg="bright_cyan")


@app.command()
def info(
    app: str = typer.Argument(..., help="Application name"),
    bid: str = typer.Argument(..., help="Defect ID")
):
    """Display defect metadata"""
    defect_info = metadata_manager.get_defect_info(app, bid)
    
    if not defect_info:
        click.secho(f"Defect not found: {app} - {bid}", fg="red")
        return
    
    # Format output (aligned two-column)
    _echo_kv_aligned([
        ("app", f"{defect_info.get('app', 'N/A')}"),
        ("repo", f"{defect_info.get('repo', 'N/A')}"),
        ("commit", f"{defect_info.get('commit', 'N/A')}"),
        ("defect_id", f"{defect_info.get('defect_id', 'N/A')}"),
        ("type", f"{defect_info.get('type', 'N/A')}"),
        ("case", f"{defect_info.get('case', 'N/A')}")
    ])
    
    # Output consequence
    consequences = defect_info.get('consequence', [])
    if consequences:
        click.secho("consequence", bold=True, fg="bright_white", nl=False)
        click.secho(":", fg="bright_black")
        for cons in consequences:
            click.secho(f"  - {cons}", fg="bright_cyan")
    
    # Output locations
    locations = defect_info.get('locations', [])
    if locations:
        click.secho("locations", bold=True, fg="bright_white", nl=False)
        click.secho(":", fg="bright_black")
        for loc in locations:
            click.secho(f"  - {loc}", fg="bright_cyan")


@app.command()
def test(
    app: str = typer.Argument(..., help="Application name"),
    bid: str = typer.Argument(..., help="Defect ID"),
    trigger: bool = typer.Option(False, "--trigger", help="Show trigger tests")
):
    """Display test information (print only, do not execute)"""
    # Get information for specific defect
    defect_info = metadata_manager.get_defect_info(app, bid)
    
    if not defect_info:
        click.secho(f"Defect not found: {app} - {bid}", fg="red")
        return
    
    if trigger:
        # Display trigger tests
        trigger_tests = defect_info.get('trigger_tests', [])
        if trigger_tests:
            click.secho("trigger_tests:", bold=True, fg="bright_white")
            for test in trigger_tests:
                if test.strip():
                    click.secho(f"- {test}", fg="bright_cyan")
        else:
            click.secho("No trigger tests available for this defect.", fg="yellow")
    else:
        # Display basic test information
        click.secho(f"Test information for {app} - {bid}", bold=True)
        click.secho(f"Defect type: {defect_info.get('type', 'N/A')}")
        click.secho(f"Case: {defect_info.get('case', 'N/A')}")
        click.secho("Use --trigger to see detailed trigger tests", fg="bright_black")


def main():
    """Main entry function"""
    app()


if __name__ == "__main__":
    main()
