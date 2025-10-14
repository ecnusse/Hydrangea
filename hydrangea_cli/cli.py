"""CLI main entry point"""

import typer
from typing import Optional
from pathlib import Path
from .core.metadata import MetadataManager
from .core.format_analyzer import FormatAnalyzer

app = typer.Typer(help="Hydrangea CLI - Query Hydrangea dataset")
metadata_manager = MetadataManager()
format_analyzer = FormatAnalyzer()


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


@app.command()
def analyze(
    app_name: str = typer.Argument(..., help="Application name to analyze")
):
    """Analyze an application for LLM input/output format issues using Comfrey framework"""
    typer.echo(f"Analyzing application: {app_name}")
    
    # Prepare report file
    report_dir = Path("repos/analyze_report")
    report_dir.mkdir(parents=True, exist_ok=True)
    
    # Handle app names with path separators
    safe_app_name = app_name.replace('/', '_').replace('\\', '_')
    report_file = report_dir / f"{safe_app_name}_analyze.txt"
    
    # Capture output for both CLI and file
    cli_output = []
    
    # Run analysis
    results = format_analyzer.analyze_application(app_name)
    
    if not results['success']:
        error_msg = f"❌ Analysis failed: {results['error']}"
        cli_output.append(error_msg)
        typer.echo(error_msg)
        return
    
    # Display results
    app_info = results['app_info']
    header = f"\n✅ Analysis completed for {app_info['app']}"
    cli_output.append(header)
    typer.echo(header)
    
    repo_info = [
        f"🔗 Repository: {app_info['url']}",
        f"📝 Commit ID: {app_info['commit_id']}",
        f"📊 Classification: {app_info['classification']}",
        f"🤖 LLM: {app_info['llm']}"
    ]
    for line in repo_info:
        cli_output.append(line)
        typer.echo(line)
    
    summary = [
        f"\n📋 Analysis Summary:",
        f"  Total files analyzed: {results['total_files_analyzed']}",
        f"  Files with format issues: {results['files_with_issues']}"
    ]
    for line in summary:
        cli_output.append(line)
        typer.echo(line)
    
    def _emit_issue_section(title: str, issues: list):
        if not issues:
            return
        cli_output.append(title)
        typer.echo(title)
        for issue in issues:
            severity = "High" if issue['severity'] > 0.7 else "Medium" if issue['severity'] > 0.3 else "Low"
            issue_info = [
                f"    - Type: {issue['type']}",
                f"      Severity: {severity} ({issue['severity']:.2f})"
            ]
            cli_output.extend(issue_info)
            for line in issue_info:
                typer.echo(line)
            violations = issue.get('details', {}).get('violations', [])
            if violations:
                count_line = f"      Violations: {len(violations)}"
                cli_output.append(count_line)
                typer.echo(count_line)
                details_lines = []
                no_preview = 0
                no_preview_positions = []
                for i, v in enumerate(violations):
                    position_info = []
                    if 'source' in v:
                        position_info.append(f"Source: {v['source']}")
                    if 'start_index' in v:
                        position_info.append(f"Index: {v['start_index']}")
                    preview = v.get('document_preview', '')
                    if preview:
                        preview_text = preview[:50] + ('...' if len(preview) > 50 else '')
                        position_text = " (" + ", ".join(position_info) + ")" if position_info else ""
                        details_lines.append(f"        * Violation {i+1}:{position_text} Preview: '{preview_text}'")
                    else:
                        no_preview += 1
                        if position_info:
                            no_preview_positions.append(f"Violation {i+1} (" + ", ".join(position_info) + ")")
                cli_output.extend(details_lines)
                if no_preview > 0:
                    if no_preview == 1 and no_preview_positions:
                        cli_output.append(f"        * {no_preview_positions[0]} - No preview available")
                    elif no_preview == 1:
                        cli_output.append(f"        * Violation {len(violations)} - No preview available")
                    else:
                        cli_output.append(f"        * {no_preview} violations - No preview available")
                        if len(no_preview_positions) > 0:
                            cli_output.append(f"          Sample positions: {', '.join(no_preview_positions[:3])}{'...' if len(no_preview_positions) > 3 else ''}")

    if results['analysis_results']:
        cli_output.append("\n🔍 Detailed Format Issues:")
        typer.echo("\n🔍 Detailed Format Issues:")
        for file_result in results['analysis_results']:
            file_line = f"\nFile: {file_result['file']}"
            cli_output.append(file_line)
            typer.echo(file_line)
            _emit_issue_section("  📝 Prompt Issues:", file_result.get('prompt_issues', []))
            _emit_issue_section("  📤 Completion Issues:", file_result.get('completion_issues', []))
    else:
        no_issues = "\n✅ No format issues detected in the analyzed files."
        cli_output.append(no_issues)
        typer.echo(no_issues)
    
    # Write to report file
    try:
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("\n".join(cli_output))
        typer.echo(f"\n📄 Analysis report saved to: {report_file}")
    except Exception as e:
        typer.echo(f"\n❌ Failed to save report: {e}")


def main():
    """Main entry function"""
    app()


if __name__ == "__main__":
    main()
