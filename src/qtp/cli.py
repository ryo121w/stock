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
@click.option("--window", "-w", default=7, help="Window size in days")
@click.option("--windows", "-n", default=4, help="Number of rolling windows")
def trend(window: int, windows: int):
    """Show accuracy trend across rolling windows to detect degradation."""
    from qtp.data.database import QTPDatabase

    database = QTPDatabase(Path("data/qtp.db"))
    console = _get_console()

    results = database.get_accuracy_trend(window_days=window, n_windows=windows)

    if not results:
        click.echo("No graded predictions found. Run: make grade")
        return

    if console:
        from rich.table import Table

        table = Table(title=f"Accuracy Trend ({window}-day windows)", show_lines=False)
        table.add_column("Window", style="dim")
        table.add_column("Predictions", justify="right")
        table.add_column("Accuracy", justify="right")
        table.add_column("Trend", justify="center")

        for i, r in enumerate(results):
            acc_str = f"{r['accuracy']:.1%}"
            if r["accuracy"] >= 0.60:
                style = "[bold green]"
            elif r["accuracy"] >= 0.55:
                style = "[green]"
            elif r["accuracy"] >= 0.50:
                style = "[yellow]"
            else:
                style = "[bold red]"

            # Trend arrow vs previous (older) window
            if i < len(results) - 1:
                prev_acc = results[i + 1]["accuracy"]
                if r["accuracy"] > prev_acc + 0.02:
                    trend_icon = "[green]^ improving[/]"
                elif r["accuracy"] < prev_acc - 0.02:
                    trend_icon = "[red]v degrading[/]"
                else:
                    trend_icon = "[dim]= stable[/]"
            else:
                trend_icon = "[dim]—[/]"

            table.add_row(
                r["window"],
                str(r["total"]),
                f"{style}{acc_str}[/]",
                trend_icon,
            )

        console.print(table)

        # Summary
        newest = results[0]["accuracy"]
        oldest = results[-1]["accuracy"]
        delta = newest - oldest
        if delta < -0.05:
            console.print(
                f"\n[bold red]WARNING: Accuracy dropped {delta:+.1%} over {len(results)} windows. "
                f"Consider retraining (make auto-retrain).[/]"
            )
        elif delta > 0.05:
            console.print(f"\n[bold green]Accuracy improving: {delta:+.1%} trend.[/]")
        else:
            console.print(f"\n[dim]Accuracy stable ({delta:+.1%} overall change).[/]")
    else:
        for r in results:
            click.echo(f"  {r['window']}: {r['accuracy']:.1%} ({r['total']} predictions)")


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


# =============================================================================
# 7-Gate Signal command
# =============================================================================


@main.command()
@click.argument("ticker")
@click.option("--config", "-c", type=str, default="configs/default.yaml")
@click.option("--market-config", "-m", type=str, default=None)
def signal(ticker: str, config: str, market_config: str | None):
    """Run 7-gate evaluation for a ticker."""
    setup_logging()
    from qtp.data.database import QTPDatabase
    from qtp.gates import GateResult

    db = QTPDatabase(Path("data/qtp.db"))
    console = _get_console()

    results: list[GateResult] = []

    # -- Gate 1: QTP quantitative model -----------------------------------
    from qtp.gates.gate1_qtp import Gate1_QTP

    g1 = Gate1_QTP(db).evaluate(ticker)
    results.append(g1)

    # -- Gate 2: Technical analysis ---------------------------------------
    try:
        from qtp.gates.gate2_technical import Gate2_Technical

        g2 = Gate2_Technical().evaluate({"ticker": ticker})
    except (ImportError, Exception) as exc:
        g2 = GateResult(gate="Technical", passed=False, score=0.0, reason=f"skipped: {exc}")
    results.append(g2)

    # -- Gate 3: Fundamental analysis -------------------------------------
    try:
        from qtp.gates.gate3_fundamental import Gate3_Fundamental

        g3 = Gate3_Fundamental().evaluate({"ticker": ticker})
    except (ImportError, Exception) as exc:
        g3 = GateResult(gate="Fundamental", passed=False, score=0.0, reason=f"skipped: {exc}")
    results.append(g3)

    # -- Gate 4: MAGI consensus -------------------------------------------
    try:
        from qtp.gates.gate4_magi import Gate4_MAGI

        g4 = Gate4_MAGI().evaluate({"ticker": ticker})
    except (ImportError, Exception) as exc:
        g4 = GateResult(gate="MAGI", passed=False, score=0.0, reason=f"skipped: {exc}")
    results.append(g4)

    # -- Gate 5: Sentiment (soft gate) ------------------------------------
    try:
        from qtp.gates.gate5_sentiment import Gate5_Sentiment

        g5 = Gate5_Sentiment().evaluate({"ticker": ticker})
    except (ImportError, Exception) as exc:
        g5 = GateResult(gate="Sentiment", passed=True, score=50.0, reason=f"skipped: {exc}")
    results.append(g5)

    # -- Gate 6: Integration (weighted composite) -------------------------
    try:
        from qtp.gates.gate6_integration import Gate6_Integration

        g6 = Gate6_Integration().evaluate(results)
    except (ImportError, Exception) as exc:
        # Fallback: simple average
        avg = sum(r.score for r in results) / max(len(results), 1)
        all_hard_passed = all(r.passed for r in results[:4])
        g6 = GateResult(
            gate="Integration",
            passed=all_hard_passed,
            score=avg,
            reason=f"fallback avg: {exc}",
        )
    results.append(g6)

    # -- Gate 7: Final verdict --------------------------------------------
    try:
        from qtp.gates.gate7_verdict import Gate7_Verdict

        g7 = Gate7_Verdict().evaluate(g6)
    except (ImportError, Exception) as exc:
        verdict = "BUY" if g6.passed and g6.score >= 70 else ("HOLD" if g6.passed else "AVOID")
        g7 = GateResult(
            gate="Verdict",
            passed=g6.passed,
            score=g6.score,
            reason=f"{verdict} (fallback: {exc})",
        )
    results.append(g7)

    # -- Display results --------------------------------------------------
    if console:
        from rich.table import Table

        table = Table(title=f"7-Gate Signal: {ticker}", show_lines=True)
        table.add_column("Gate", style="bold")
        table.add_column("Pass", justify="center")
        table.add_column("Score", justify="right")
        table.add_column("Reason")
        table.add_column("Warnings", style="dim")

        for r in results:
            passed_str = "[bold green]PASS[/]" if r.passed else "[bold red]FAIL[/]"
            score_str = f"{r.score:.0f}"
            if r.score >= 70:
                score_str = f"[green]{score_str}[/]"
            elif r.score >= 50:
                score_str = f"[yellow]{score_str}[/]"
            else:
                score_str = f"[red]{score_str}[/]"
            warnings = "; ".join(r.warnings) if r.warnings else ""
            table.add_row(r.gate, passed_str, score_str, r.reason, warnings)

        console.print(table)

        # Final verdict banner
        final = results[-1]
        if final.passed and final.score >= 70:
            console.print(
                f"\n[bold green]>>> SIGNAL: BUY {ticker} (score={final.score:.0f}) <<<[/]"
            )
        elif final.passed:
            console.print(
                f"\n[bold yellow]>>> SIGNAL: HOLD {ticker} (score={final.score:.0f}) <<<[/]"
            )
        else:
            console.print(
                f"\n[bold red]>>> SIGNAL: AVOID {ticker} (score={final.score:.0f}) <<<[/]"
            )
    else:
        click.echo(f"\n7-Gate Signal: {ticker}")
        click.echo("=" * 50)
        for r in results:
            status = "PASS" if r.passed else "FAIL"
            click.echo(f"  {r.gate:15s} [{status}] score={r.score:.0f}  {r.reason}")
            for w in r.warnings:
                click.echo(f"    WARNING: {w}")
        final = results[-1]
        verdict = (
            "BUY" if final.passed and final.score >= 70 else ("HOLD" if final.passed else "AVOID")
        )
        click.echo(f"\n  >>> {verdict} (score={final.score:.0f}) <<<")


if __name__ == "__main__":
    main()
