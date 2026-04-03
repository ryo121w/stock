#!/bin/bash
# Daily prediction job — run via crontab
# crontab -e → 0 9 * * 1-5 /Users/wakiryoutarou/quant-trading-pipeline/scripts/daily_predict.sh

set -e
cd /Users/wakiryoutarou/quant-trading-pipeline

VENV=".venv/bin"
DATE=$(date +%Y-%m-%d)
LOG="data/reports/daily_${DATE}.log"

echo "=== QTP Daily Prediction: ${DATE} ===" | tee "$LOG"

# 1. Fetch latest data
echo "[1/5] Fetching OHLCV..." | tee -a "$LOG"
$VENV/qtp fetch >> "$LOG" 2>&1

# 1.5 Fetch MCP alternative data (daily accumulation for Tier5 time-series)
echo "[1.5/5] Fetching MCP alternative data..." | tee -a "$LOG"
for TICKER in MSFT GOOGL AMZN NVDA META JPM V; do
  $VENV/python -c "
from qtp.data.fetchers.mcp_alternative import fetch_alternative_data
from qtp.data.database import QTPDatabase
from pathlib import Path
from datetime import datetime

db = QTPDatabase(Path('data/qtp.db'))
today = datetime.now().strftime('%Y-%m-%d')

# Fetch via MCP and save to legacy table
results = fetch_alternative_data('$TICKER', db, force_refresh=True)

# Also save to daily accumulation table
for tool_name, data in results.items():
    cache_ticker = '_market' if tool_name == 'market_regime' else '$TICKER'
    db.upsert_alternative_daily(cache_ticker, tool_name, data, date=today)

print(f'  $TICKER: {len(results)} tools fetched')
" >> "$LOG" 2>&1
done
echo "  MCP data saved to alternative_data_daily" | tee -a "$LOG"

# 2. Quick train
echo "[2/5] Training (fast)..." | tee -a "$LOG"
$VENV/qtp train -m configs/phase3_best.yaml --fast >> "$LOG" 2>&1

# 3. Predict
echo "[3/5] Generating predictions..." | tee -a "$LOG"
$VENV/qtp predict -m configs/phase3_best.yaml >> "$LOG" 2>&1

# 4. Save to JSON and push
echo "[4/5] Saving and pushing..." | tee -a "$LOG"
mkdir -p predictions
$VENV/python -c "
from pathlib import Path
from qtp.data.database import QTPDatabase
import json
db = QTPDatabase(Path('data/qtp.db'))
preds = db.get_recent_predictions(10)
Path('predictions/${DATE}.json').write_text(
    json.dumps({'date': '${DATE}', 'predictions': preds}, indent=2, default=str)
)
print(f'Saved {len(preds)} predictions')
" | tee -a "$LOG"

git add predictions/
git commit -m "daily prediction: ${DATE}" || true
git push origin main || true

echo "=== Done ===" | tee -a "$LOG"
