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
echo "[1/4] Fetching OHLCV..." | tee -a "$LOG"
$VENV/qtp fetch >> "$LOG" 2>&1

# 2. Quick train
echo "[2/4] Training (fast)..." | tee -a "$LOG"
$VENV/qtp train -m configs/phase2_experiment.yaml --fast >> "$LOG" 2>&1

# 3. Predict
echo "[3/4] Generating predictions..." | tee -a "$LOG"
$VENV/qtp predict -m configs/phase2_experiment.yaml >> "$LOG" 2>&1

# 4. Save to JSON and push
echo "[4/4] Saving and pushing..." | tee -a "$LOG"
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
