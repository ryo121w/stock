#!/bin/bash
# Weekly accuracy check + auto-retrain if needed
# Add to crontab: 0 10 * * 6 /path/to/auto_retrain.sh

set -e
cd /Users/wakiryoutarou/quant-trading-pipeline
VENV=".venv/bin"
DATE=$(date +%Y-%m-%d)
LOG="data/reports/retrain_check_${DATE}.log"

mkdir -p data/reports

echo "=== Retrain Check: ${DATE} ===" | tee "$LOG"

# 1. Grade any ungraded predictions first
echo "[1/3] Grading predictions..." | tee -a "$LOG"
$VENV/python scripts/grade_predictions.py >> "$LOG" 2>&1 || true

# 2. Check recent accuracy
echo "[2/3] Checking accuracy..." | tee -a "$LOG"
RESULT=$($VENV/python -c "
from qtp.data.database import QTPDatabase
from pathlib import Path
db = QTPDatabase(Path('data/qtp.db'))
s = db.get_accuracy_summary(days=30)
total = s.get('total', 0)
if total < 10:
    print('INSUFFICIENT')
elif s.get('accuracy', 0) < 0.55:
    print(f'RETRAIN:{s[\"accuracy\"]:.3f}')
else:
    print(f'OK:{s[\"accuracy\"]:.3f}')
")

echo "  Result: $RESULT" | tee -a "$LOG"

# 3. Retrain if needed
case "$RESULT" in
    RETRAIN*)
        ACCURACY=$(echo "$RESULT" | cut -d: -f2)
        echo "[3/3] Accuracy $ACCURACY < 55%, RETRAINING..." | tee -a "$LOG"
        $VENV/qtp fetch >> "$LOG" 2>&1
        $VENV/qtp train -m configs/phase3_best.yaml >> "$LOG" 2>&1
        echo "  Retrain complete." | tee -a "$LOG"

        # Push new model
        git add data/reports/
        git commit -m "auto-retrain: accuracy $ACCURACY triggered retrain on $DATE" || true
        git push origin main || true
        ;;
    OK*)
        ACCURACY=$(echo "$RESULT" | cut -d: -f2)
        echo "[3/3] Accuracy $ACCURACY >= 55%, no retrain needed." | tee -a "$LOG"
        ;;
    INSUFFICIENT*)
        echo "[3/3] Not enough graded predictions (<10). Skipping." | tee -a "$LOG"
        ;;
esac

echo "=== Done ===" | tee "$LOG"
