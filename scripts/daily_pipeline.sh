#!/bin/bash
# Daily QTP pipeline: fetch → predict → grade → alt-data
# Run via cron: 0 9 * * 1-5 /path/to/daily_pipeline.sh
#
# Schedule (JST):
#   09:00 — US market closed, JP market open
#   This captures yesterday's US close + today's JP morning

set -euo pipefail

PROJECT_DIR="/Users/wakiryoutarou/quant-trading-pipeline"

# Load .env if exists (for FINNHUB_API_KEY etc.)
if [ -f "${PROJECT_DIR}/.env" ]; then
    set -a; source "${PROJECT_DIR}/.env"; set +a
fi
VENV_PYTHON="${PROJECT_DIR}/.venv/bin/python"
CONFIG="${PROJECT_DIR}/configs/phase5_optimized.yaml"
LOG_DIR="${PROJECT_DIR}/data/logs"
LOG_FILE="${LOG_DIR}/daily_$(date +%Y%m%d).log"

mkdir -p "${LOG_DIR}"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "${LOG_FILE}"
}

log "=== Daily Pipeline Start ==="

# Step 1: Fetch latest OHLCV
log "Step 1: Fetching OHLCV data..."
cd "${PROJECT_DIR}" && ${VENV_PYTHON} -m qtp fetch -c "${CONFIG}" >> "${LOG_FILE}" 2>&1
log "Step 1: Done"

# Step 2: Generate predictions
log "Step 2: Generating predictions..."
cd "${PROJECT_DIR}" && ${VENV_PYTHON} -m qtp predict -c "${CONFIG}" >> "${LOG_FILE}" 2>&1
log "Step 2: Done"

# Step 3: Grade past predictions
log "Step 3: Grading past predictions..."
cd "${PROJECT_DIR}" && ${VENV_PYTHON} -m qtp grade >> "${LOG_FILE}" 2>&1
log "Step 3: Done"

# Step 4: Fetch alternative data (EDGAR + Fear&Greed + Finnhub)
log "Step 4: Fetching alternative data..."
cd "${PROJECT_DIR}" && ${VENV_PYTHON} scripts/daily_alt_data.py >> "${LOG_FILE}" 2>&1
log "Step 4: Done"

# Step 5: Show summary
log "Step 5: Summary"
cd "${PROJECT_DIR}" && ${VENV_PYTHON} -c "
import sqlite3
conn = sqlite3.connect('data/qtp.db')
r = conn.execute('SELECT COUNT(*) FROM predictions WHERE graded_at IS NULL').fetchone()
r2 = conn.execute('SELECT COUNT(*) FROM predictions WHERE graded_at IS NOT NULL').fetchone()
print(f'Predictions: {r2[0]} graded, {r[0]} ungraded')

# Today's predictions
today_preds = conn.execute('''
    SELECT ticker, direction, confidence
    FROM predictions
    WHERE prediction_date = date(\"now\")
    ORDER BY confidence DESC
''').fetchall()
if today_preds:
    print('Today\\'s signals:')
    for t, d, c in today_preds:
        signal = 'BUY' if d == 1 and c >= 0.55 else ('SELL' if d == 0 and c >= 0.55 else 'HOLD')
        print(f'  {t:10s} {\"UP\" if d == 1 else \"DN\"} {c:.1%} → {signal}')
conn.close()
" >> "${LOG_FILE}" 2>&1

log "=== Daily Pipeline Complete ==="
