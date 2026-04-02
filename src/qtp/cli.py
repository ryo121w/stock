"""CLI entry point for the Quant Trading Pipeline."""

from __future__ import annotations

from pathlib import Path

import click

from qtp.config import PipelineConfig
from qtp.utils.logging_ import setup_logging


def _load_config(config_path: str, market_config: str | None = None) -> PipelineConfig:
    paths = [Path(config_path)]
    if market_config:
        paths.append(Path(market_config))
    return PipelineConfig.from_yamls(*paths)


def _get_console():
    """Lazy import rich console."""
    try:
        from rich.console import Console

        return Console()
    except ImportError:
        return None


def _print_predictions(predictions):
    """Print predictions with rich table if available, else plain text."""
    console = _get_console()
    if console and predictions:
        from rich.table import Table

        table = Table(title="Predictions", show_lines=False)
        table.add_column("Ticker", style="bold")
        table.add_column("Dir", justify="center")
        table.add_column("Confidence", justify="right")
        table.add_column("Magnitude", justify="right")
        table.add_column("Signal", justify="center")

        for p in predictions:
            direction = "[green]UP[/]" if p.direction == 1 else "[red]DOWN[/]"
            conf = f"{p.direction_proba:.1%}"
            mag = f"{p.magnitude:+.2%}"
            if p.direction_proba >= 0.55:
                signal = "[bold green]BUY[/]" if p.direction == 1 else "[bold red]SELL[/]"
            else:
                signal = "[dim]HOLD[/]"
            table.add_row(p.ticker, direction, conf, mag, signal)

        console.print(table)
    else:
        for p in predictions:
            emoji = "\u2b06" if p.direction == 1 else "\u2b07"
            click.echo(
                f"  {emoji} {p.ticker}: {p.direction_proba:.1%} conf, {p.magnitude:+.2%} mag"
            )


# =============================================================================
# Main CLI group
# =============================================================================


@click.group()
@click.version_option(package_name="quant-trading-pipeline")
def main():
    """Quantitative Trading Pipeline CLI."""
    pass


# =============================================================================
# Pipeline commands
# =============================================================================


@main.command()
@click.option("--config", "-c", type=str, default="configs/default.yaml")
@click.option(
    "--market-config",
    "-m",
    type=str,
    default=None,
    help="Additional market config (e.g., configs/jp_stocks.yaml)",
)
def fetch(config: str, market_config: str | None):
    """Fetch and store OHLCV data for the configured universe."""
    setup_logging()
    cfg = _load_config(config, market_config)
    from qtp.pipeline import PipelineRunner

    runner = PipelineRunner(cfg)
    runner.run_fetch()
    click.echo("Fetch complete.")


@main.command()
@click.option("--config", "-c", type=str, default="configs/default.yaml")
@click.option("--market-config", "-m", type=str, default=None)
@click.option("--fast", is_flag=True, help="Fast mode: limit WF CV to 3 folds for quick iteration")
def train(config: str, market_config: str | None, fast: bool):
    """Train model with cross-validation."""
    setup_logging()
    cfg = _load_config(config, market_config)
    from qtp.pipeline import PipelineRunner

    runner = PipelineRunner(cfg)
    version = runner.run_train(fast=fast)

    console = _get_console()
    if console:
        console.print(f"\n[bold green]Training complete.[/] Model: [cyan]{version}[/]")
    else:
        click.echo(f"Training complete. Model version: {version}")


@main.command()
@click.option("--config", "-c", type=str, default="configs/default.yaml")
@click.option("--market-config", "-m", type=str, default=None)
@click.option(
    "--version", "-v", type=str, default=None, help="Model version to use (default: latest)"
)
def predict(config: str, market_config: str | None, version: str | None):
    """Generate predictions for today."""
    setup_logging()
    cfg = _load_config(config, market_config)
    from qtp.pipeline import PipelineRunner

    runner = PipelineRunner(cfg)
    predictions = runner.run_predict(version)
    _print_predictions(predictions)


@main.command(name="run-all")
@click.option("--config", "-c", type=str, default="configs/default.yaml")
@click.option("--market-config", "-m", type=str, default=None)
@click.option("--fast", is_flag=True, help="Fast mode: limit WF CV to 3 folds")
def run_all(config: str, market_config: str | None, fast: bool):
    """Full pipeline: fetch -> train -> predict."""
    setup_logging()
    cfg = _load_config(config, market_config)
    from qtp.pipeline import PipelineRunner

    runner = PipelineRunner(cfg)
    result = runner.run_all(fast=fast)

    console = _get_console()
    if console:
        console.print(
            f"\n[bold green]Pipeline complete.[/] Model: [cyan]{result['model_version']}[/]"
        )
    else:
        click.echo(f"\nPipeline complete. Model: {result['model_version']}")

    _print_predictions(result["predictions"])


# =============================================================================
# Database / experiment commands
# =============================================================================


@main.group()
def db():
    """Database commands: inspect models, experiments, and alternative data."""
    pass


@db.command()
def status():
    """Show database status overview."""
    from qtp.data.database import QTPDatabase

    database = QTPDatabase(Path("data/qtp.db"))
    console = _get_console()

    models = database.list_models(limit=5)
    exps = database.list_experiments(limit=5)
    coverage = database.alternative_coverage()

    if console:
        from rich.table import Table

        # Models
        mt = Table(title="Recent Models", show_lines=False)
        mt.add_column("Version", style="cyan")
        mt.add_column("Type")
        mt.add_column("Created")
        for m in models:
            mt.add_row(m["version"], m["model_type"] or "?", m["created_at"] or "?")
        console.print(mt)

        # Experiments
        et = Table(title="Recent Experiments", show_lines=False)
        et.add_column("ID", style="bold")
        et.add_column("Horizon")
        et.add_column("Threshold")
        et.add_column("WF AUC", justify="right", style="green")
        et.add_column("WF Sharpe", justify="right")
        et.add_column("Win Rate", justify="right")
        et.add_column("Created")
        for e in exps:
            et.add_row(
                str(e["id"]),
                str(e.get("label_horizon", "?")),
                str(e.get("label_threshold", "?")),
                f"{e['wf_auc']:.4f}" if e.get("wf_auc") else "?",
                f"{e['wf_sharpe']:.2f}" if e.get("wf_sharpe") else "?",
                f"{e['wf_win_rate']:.1%}" if e.get("wf_win_rate") else "?",
                e.get("created_at", "?"),
            )
        console.print(et)

        # Alt data coverage
        if coverage:
            at = Table(title="Alternative Data Coverage", show_lines=False)
            at.add_column("Ticker", style="cyan")
            at.add_column("Tools", justify="right")
            at.add_column("Last Updated")
            for c in coverage:
                at.add_row(c["ticker"], str(c["n_tools"]), c.get("newest", "?"))
            console.print(at)
        else:
            console.print("[dim]No alternative data cached yet.[/]")
    else:
        click.echo(f"Models: {len(models)}")
        for m in models:
            click.echo(f"  {m['version']} ({m['created_at']})")
        click.echo(f"Experiments: {len(exps)}")
        for e in exps:
            click.echo(f"  #{e['id']} AUC={e.get('wf_auc')} h={e.get('label_horizon')}")
        click.echo(f"Alt data: {len(coverage)} tickers")


@db.command()
@click.option("--metric", default="wf_auc", help="Metric to rank by")
@click.option("--limit", "-n", default=10, help="Number of results")
def best(metric: str, limit: int):
    """Show best experiments ranked by a metric."""
    from qtp.data.database import QTPDatabase

    database = QTPDatabase(Path("data/qtp.db"))
    console = _get_console()

    try:
        results = database.best_experiments(metric, limit)
    except ValueError as e:
        click.echo(f"Error: {e}")
        return

    if console:
        from rich.table import Table

        table = Table(title=f"Top {limit} Experiments by {metric}", show_lines=False)
        table.add_column("Rank", style="bold")
        table.add_column("ID")
        table.add_column("Horizon")
        table.add_column("Threshold")
        table.add_column("Tiers")
        table.add_column("WF AUC", justify="right", style="green")
        table.add_column("WF Sharpe", justify="right")
        table.add_column("Win Rate", justify="right")
        table.add_column("Folds", justify="right")
        table.add_column("Model")

        for i, e in enumerate(results, 1):
            table.add_row(
                str(i),
                str(e["id"]),
                str(e.get("label_horizon", "?")),
                str(e.get("label_threshold", "?")),
                str(e.get("feature_tiers", "?")),
                f"{e['wf_auc']:.4f}" if e.get("wf_auc") else "?",
                f"{e['wf_sharpe']:.2f}" if e.get("wf_sharpe") else "?",
                f"{e['wf_win_rate']:.1%}" if e.get("wf_win_rate") else "?",
                str(e.get("wf_n_folds", "?")),
                e.get("model_version", "?"),
            )
        console.print(table)
    else:
        for i, e in enumerate(results, 1):
            click.echo(
                f"  #{i} AUC={e.get('wf_auc'):.4f} Sharpe={e.get('wf_sharpe'):.2f} "
                f"h={e.get('label_horizon')} t={e.get('label_threshold')}"
            )


@db.command()
def stale():
    """Show alternative data that needs refreshing (>24h old)."""
    from qtp.data.database import QTPDatabase

    database = QTPDatabase(Path("data/qtp.db"))
    console = _get_console()

    stale_data = database.list_stale_data(max_age_hours=24)

    if console:
        if stale_data:
            from rich.table import Table

            table = Table(title="Stale Alternative Data (>24h)", show_lines=False)
            table.add_column("Ticker", style="cyan")
            table.add_column("Tool")
            table.add_column("Last Fetched", style="red")
            for s in stale_data:
                table.add_row(s["ticker"], s["tool"], s["fetched_at"])
            console.print(table)
        else:
            console.print("[green]All alternative data is fresh.[/]")
    else:
        if stale_data:
            for s in stale_data:
                click.echo(f"  {s['ticker']}/{s['tool']} — last: {s['fetched_at']}")
        else:
            click.echo("All data is fresh.")


@db.command()
def predictions():
    """Show recent predictions and grading results."""
    from qtp.data.database import QTPDatabase

    database = QTPDatabase(Path("data/qtp.db"))
    console = _get_console()

    recent = database.get_recent_predictions(limit=20)
    ungraded = database.get_ungraded_predictions()

    if console:
        from rich.table import Table

        table = Table(title=f"Recent Predictions ({len(ungraded)} ungraded)", show_lines=False)
        table.add_column("Date", style="dim")
        table.add_column("Ticker", style="cyan bold")
        table.add_column("Dir", justify="center")
        table.add_column("Conf", justify="right")
        table.add_column("Actual", justify="right")
        table.add_column("Result", justify="center")

        for p in recent:
            direction = "[green]UP[/]" if p["direction"] == 1 else "[red]DOWN[/]"
            conf = f"{p['confidence']:.1%}"
            if p.get("actual_return") is not None:
                actual = f"{p['actual_return']:+.2%}"
                result = "[bold green]OK[/]" if p.get("is_correct") else "[bold red]NG[/]"
            else:
                actual = "[dim]—[/]"
                result = "[dim]pending[/]"
            table.add_row(str(p["prediction_date"]), p["ticker"], direction, conf, actual, result)

        console.print(table)
    else:
        for p in recent:
            status = "OK" if p.get("is_correct") else ("NG" if p.get("is_correct") == 0 else "?")
            click.echo(
                f"  {p['prediction_date']} {p['ticker']} "
                f"{'UP' if p['direction'] == 1 else 'DN'} {p['confidence']:.1%} → {status}"
            )


@db.command()
def accuracy():
    """Show prediction accuracy report."""
    from qtp.data.database import QTPDatabase

    database = QTPDatabase(Path("data/qtp.db"))
    console = _get_console()

    summary = database.get_accuracy_summary()
    if not summary or not summary.get("total"):
        click.echo("No graded predictions yet. Run: make grade")
        return

    if console:
        from rich.table import Table

        console.print(
            f"\n[bold]Overall Accuracy: {summary['accuracy']:.1%}[/] "
            f"({summary['correct']}/{summary['total']})"
        )
        if summary.get("avg_return"):
            console.print(f"Avg Return: {summary['avg_return']:+.3%}")

        # By confidence
        by_conf = database.get_accuracy_by_confidence()
        if by_conf:
            t = Table(title="Accuracy by Confidence", show_lines=False)
            t.add_column("Bucket")
            t.add_column("Total", justify="right")
            t.add_column("Accuracy", justify="right", style="green")
            t.add_column("Avg Return", justify="right")
            for b in by_conf:
                t.add_row(
                    b["bucket"], str(b["total"]), f"{b['accuracy_pct']}%", f"{b['avg_return_pct']}%"
                )
            console.print(t)

        # By ticker
        by_ticker = database.get_accuracy_by_ticker()
        if by_ticker:
            t = Table(title="Accuracy by Ticker", show_lines=False)
            t.add_column("Ticker", style="cyan")
            t.add_column("Total", justify="right")
            t.add_column("Accuracy", justify="right", style="green")
            t.add_column("Avg Return", justify="right")
            for b in by_ticker:
                t.add_row(
                    b["ticker"], str(b["total"]), f"{b['accuracy_pct']}%", f"{b['avg_return_pct']}%"
                )
            console.print(t)
    else:
        click.echo(f"Accuracy: {summary['accuracy']:.1%} ({summary['correct']}/{summary['total']})")


@main.command()
def grade():
    """Grade past predictions with actual prices."""
    setup_logging()
    import subprocess
    import sys

    result = subprocess.run(
        [sys.executable, "scripts/grade_predictions.py"],
        capture_output=False,
    )
    raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
