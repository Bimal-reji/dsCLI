"""Rich console helpers shared by CLI commands.

Keeps the visual language of dscli consistent: green checkmarks for success,
red for errors, tables for structured output, and panels for summaries.
"""

from __future__ import annotations

from typing import Any, Iterable

from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from dscli.evaluation.metrics import METRIC_LABELS

console = Console(highlight=False)


def print_success(message: str) -> None:
    console.print(f"[bold green]✓[/bold green] {message}")


def print_error(message: str) -> None:
    console.print(f"[bold red]✗[/bold red] {message}")


def print_warning(message: str) -> None:
    console.print(f"[bold yellow]![/bold yellow] {message}")


def print_info(message: str) -> None:
    console.print(f"[bold cyan]›[/bold cyan] {message}")


def success_panel(title: str, message: str) -> None:
    console.print(Panel(f"[green]{message}[/green]", title=f"[bold]{title}[/bold]", border_style="green", expand=False))


def error_panel(message: str) -> None:
    console.print(
        Panel(message, title="[bold]Error[/bold]", border_style="red", expand=False)
    )


def metrics_table(metrics: dict[str, Any], title: str = "Model Performance") -> Table:
    """Build a Rich table of flat metrics (structured values are skipped)."""
    table = Table(title=title, box=box.ROUNDED, title_style="bold")
    table.add_column("Metric", style="cyan")
    table.add_column("Score", justify="right")
    for key, value in metrics.items():
        if not isinstance(value, (int, float)):
            continue
        label = METRIC_LABELS.get(key, key.replace("_", " ").title())
        table.add_row(label, f"{value:.4f}")
    return table


def summary_table(title: str, rows: Iterable[tuple[str, str]]) -> Table:
    """Build a two-column key/value table."""
    table = Table(title=title, box=box.SIMPLE_HEAVY, title_style="bold")
    table.add_column("Key", style="cyan", no_wrap=True)
    table.add_column("Value")
    for key, value in rows:
        table.add_row(key, str(value))
    return table
