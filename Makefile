.PHONY: help setup install test test-all test-slow test-parallel lint format typecheck check train train-fast run run-fast dashboard clean

PYTHON := .venv/bin/python
PYTEST := .venv/bin/python -m pytest
RUFF := .venv/bin/ruff
MYPY := .venv/bin/mypy
QTP := .venv/bin/qtp

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

# ── Setup ────────────────────────────────────────────────────────────
setup: install hooks ## Full dev setup (install + pre-commit hooks)

install: ## Install package in dev mode
	uv pip install --python .venv/bin/python -e ".[dev]"

hooks: ## Install pre-commit hooks
	.venv/bin/pre-commit install

# ── Testing ──────────────────────────────────────────────────────────
test: ## Run fast tests only (~1s)
	$(PYTEST) -v --tb=short

test-all: ## Run ALL tests including slow (~5s)
	$(PYTEST) -v --tb=short -m ""

test-slow: ## Run only slow tests (model training)
	$(PYTEST) -v --tb=short -m "slow"

test-parallel: ## Run all tests in parallel
	$(PYTEST) -v --tb=short -m "" -n auto

test-cov: ## Run tests with coverage report
	$(PYTEST) -v --tb=short -m "" --cov=qtp --cov-report=term-missing

# ── Code Quality ─────────────────────────────────────────────────────
lint: ## Check code style (ruff)
	$(RUFF) check src/ tests/ scripts/

format: ## Auto-format code (ruff)
	$(RUFF) format src/ tests/ scripts/
	$(RUFF) check --fix src/ tests/ scripts/

typecheck: ## Run mypy type checking
	$(MYPY) src/qtp/ --ignore-missing-imports

check: lint typecheck test ## Full quality gate: lint + types + test

# ── Pipeline ─────────────────────────────────────────────────────────
fetch: ## Fetch OHLCV data
	$(QTP) fetch

train: ## Train with full Walk-Forward CV (~2-5 min)
	$(QTP) train

train-fast: ## Train with 3 WF folds only (~5s)
	$(QTP) train --fast

run: ## Full pipeline: fetch -> train -> predict
	$(QTP) run-all

run-fast: ## Full pipeline in fast mode
	$(QTP) run-all --fast

predict: ## Generate predictions with latest model
	$(QTP) predict

# ── Experiments ──────────────────────────────────────────────────────
train-p2: ## Train Phase 2 (horizon=5, threshold=2%)
	$(QTP) train -m configs/phase2_experiment.yaml

train-p2-fast: ## Phase 2 fast mode
	$(QTP) train -m configs/phase2_experiment.yaml --fast

baseline: ## Run honest baseline comparison
	$(PYTHON) scripts/honest_baseline.py

alpha: ## Run feature alpha test
	$(PYTHON) scripts/alpha_test.py

# ── Database ─────────────────────────────────────────────────────────
db-status: ## Show DB status (rich tables)
	$(QTP) db status

db-best: ## Show best experiments
	$(QTP) db best

db-stale: ## Show stale alternative data
	$(QTP) db stale

db-predictions: ## Show recent predictions
	$(QTP) db predictions

db-accuracy: ## Show prediction accuracy report
	$(QTP) db accuracy

db-trend: ## Show accuracy trend (degradation detection)
	$(QTP) db trend

# ── Prediction Grading ───────────────────────────────────────────────
grade: ## Grade past predictions with actual prices
	$(PYTHON) scripts/grade_predictions.py

grade-report: ## Show accuracy report only (no grading)
	$(PYTHON) scripts/grade_predictions.py --report

backfill: ## Backfill historical predictions + grade immediately
	$(PYTHON) scripts/backfill_predictions.py

auto-retrain: ## Check accuracy + auto-retrain if degraded (<55%)
	bash scripts/auto_retrain.sh

# ── Dashboard ────────────────────────────────────────────────────────
dashboard: ## Launch Streamlit dashboard
	$(PYTHON) -m streamlit run dashboard/app.py

# ── Cleanup ──────────────────────────────────────────────────────────
clean: ## Remove caches and build artifacts
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .mypy_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .ruff_cache -exec rm -rf {} + 2>/dev/null || true
	rm -rf dist/ build/ *.egg-info/ htmlcov/ .coverage
