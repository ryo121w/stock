"""CLI entry point for the Quant Trading Pipeline."""

from __future__ import annotations

from pathlib import Path

import click

from qtp.config import PipelineConfig
from qtp.utils.logging_ import setup_logging


@click.group()
def main():
    """Quantitative Trading Pipeline CLI."""
    pass


def _load_config(config_path: str, market_config: str | None = None) -> PipelineConfig:
    paths = [Path(config_path)]
    if market_config:
        paths.append(Path(market_config))
    return PipelineConfig.from_yamls(*paths)


@main.command()
@click.option("--config", "-c", type=str, default="configs/default.yaml")
@click.option("--market-config", "-m", type=str, default=None,
              help="Additional market config (e.g., configs/jp_stocks.yaml)")
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
def train(config: str, market_config: str | None):
    """Train model with cross-validation."""
    setup_logging()
    cfg = _load_config(config, market_config)
    from qtp.pipeline import PipelineRunner
    runner = PipelineRunner(cfg)
    version = runner.run_train()
    click.echo(f"Training complete. Model version: {version}")


@main.command()
@click.option("--config", "-c", type=str, default="configs/default.yaml")
@click.option("--market-config", "-m", type=str, default=None)
@click.option("--version", "-v", type=str, default=None,
              help="Model version to use (default: latest)")
def predict(config: str, market_config: str | None, version: str | None):
    """Generate predictions for today."""
    setup_logging()
    cfg = _load_config(config, market_config)
    from qtp.pipeline import PipelineRunner
    runner = PipelineRunner(cfg)
    predictions = runner.run_predict(version)
    for p in predictions:
        emoji = "\u2b06" if p.direction == 1 else "\u2b07"
        click.echo(f"  {emoji} {p.ticker}: {p.direction_proba:.1%} conf, {p.magnitude:+.2%} mag")


@main.command(name="run-all")
@click.option("--config", "-c", type=str, default="configs/default.yaml")
@click.option("--market-config", "-m", type=str, default=None)
def run_all(config: str, market_config: str | None):
    """Full pipeline: fetch → train → predict."""
    setup_logging()
    cfg = _load_config(config, market_config)
    from qtp.pipeline import PipelineRunner
    runner = PipelineRunner(cfg)
    result = runner.run_all()
    click.echo(f"\nPipeline complete. Model: {result['model_version']}")
    click.echo(f"Predictions: {len(result['predictions'])}")
    for p in result["predictions"]:
        emoji = "\u2b06" if p.direction == 1 else "\u2b07"
        click.echo(f"  {emoji} {p.ticker}: {p.direction_proba:.1%} conf, {p.magnitude:+.2%} mag")


if __name__ == "__main__":
    main()
