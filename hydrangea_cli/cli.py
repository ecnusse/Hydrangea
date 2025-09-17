"""CLI main entry point"""

import typer
from typing import Optional
from .core.metadata import MetadataManager

app = typer.Typer(help="Hydrangea CLI - Query Hydrangea dataset")
metadata_manager = MetadataManager()


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
        typer.echo("No applications found.")
        return
    
    for app_name in apps_list:
        typer.echo(app_name)


@app.command()
def bids(
    app: Optional[str] = typer.Option(None, "--app", help="Filter defect IDs by application")
):
    """List all defect IDs"""
    defect_ids = metadata_manager.get_defect_ids(app=app)
    
    if not defect_ids:
        typer.echo("No defect IDs found.")
        return
    
    for defect_id in defect_ids:
        typer.echo(defect_id)


@app.command()
def info(
    app: str = typer.Argument(..., help="Application name"),
    bid: str = typer.Argument(..., help="Defect ID")
):
    """Display defect metadata"""
    defect_info = metadata_manager.get_defect_info(app, bid)
    
    if not defect_info:
        typer.echo(f"Defect not found: {app} - {bid}")
        return
    
    # Format output
    typer.echo(f"app: {defect_info.get('app', 'N/A')}")
    typer.echo(f"repo: {defect_info.get('repo', 'N/A')}")
    typer.echo(f"commit: {defect_info.get('commit', 'N/A')}")
    typer.echo(f"defect_id: {defect_info.get('defect_id', 'N/A')}")
    typer.echo(f"type: {defect_info.get('type', 'N/A')}")
    typer.echo(f"case: {defect_info.get('case', 'N/A')}")
    
    # Output consequence
    consequences = defect_info.get('consequence', [])
    if consequences:
        typer.echo("consequence:")
        for cons in consequences:
            typer.echo(f"  - {cons}")
    
    # Output locations
    locations = defect_info.get('locations', [])
    if locations:
        typer.echo("locations:")
        for loc in locations:
            typer.echo(f"  - {loc}")


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
        typer.echo(f"Defect not found: {app} - {bid}")
        return
    
    if trigger:
        # Display trigger tests
        trigger_tests = defect_info.get('trigger_tests', [])
        if trigger_tests:
            typer.echo("trigger_tests:")
            for test in trigger_tests:
                if test.strip():
                    typer.echo(f"- {test}")
        else:
            typer.echo("No trigger tests available for this defect.")
    else:
        # Display basic test information
        typer.echo(f"Test information for {app} - {bid}")
        typer.echo(f"Defect type: {defect_info.get('type', 'N/A')}")
        typer.echo(f"Case: {defect_info.get('case', 'N/A')}")
        typer.echo("Use --trigger to see detailed trigger tests")


def main():
    """Main entry function"""
    app()


if __name__ == "__main__":
    main()
