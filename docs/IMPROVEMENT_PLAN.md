# QTP Improvement Plan — 全改善案詳細計画

## 現状ベースライン (2026-04-03 時点)

| 指標 | 値 |
|------|-----|
| conf 55%+ 精度 | 61.3% (2,367件) |
| conf 50%+ 精度 | 63.0% (2,698件) |
| 平均リターン (55%+) | +1.92% / 10日 |
| モデル | XGB 63% + LGBM 37% |
| 特徴量 | 26個 (Top20 + Tier5時系列6) |
| 銘柄 | 7 (MSFT, GOOGL, AMZN, NVDA, META, JPM, V) |
| ラベル | h=10日, threshold=3% |
| Backtest Sharpe | 2.76 (conf_65) |

---

## Phase 4A: 低コスト高インパクト（難易度: 低）

### A3. Tier5 日次データ蓄積

**目的**: 静的スナップショットだったMCPデータを日次で蓄積し、時系列学習を可能にする

**現状の問題**:
- EPS改定、アナリスト変更は日々変化するが、1回取得した値を全期間にコピーしている
- 「EPS改定が3日連続上方修正 → 株価上昇」のパターンを学習できない

**実装計画**:

1. `scripts/daily_predict.sh` にMCPデータ取得を追加:
```bash
# 既存の fetch → train → predict の前に
for TICKER in MSFT GOOGL AMZN NVDA META JPM V; do
  $VENV/python -c "
from qtp.data.fetchers.mcp_alternative import fetch_alternative_data
from qtp.data.database import QTPDatabase
from pathlib import Path
db = QTPDatabase(Path('data/qtp.db'))
fetch_alternative_data('$TICKER', db)
"
done
```

2. `alternative_data` テーブルのスキーマ変更:
   - 現在: UNIQUE(ticker, tool) → 最新1件のみ保持
   - 変更: UNIQUE(ticker, tool, date) → 日次で蓄積
   - マイグレーション: `ALTER TABLE` + 新インデックス

3. `tier5_alternative.py` を時系列対応に改修:
   - `_load_alt()` が日付を受け取り、その日付以前の最新レコードを返す
   - `_load_alt_timeseries()` を追加: 過去N日分のデータを取得し、変化率を特徴量化
   - 新特徴量:
     - `eps_revision_momentum_7d`: 過去7日間のEPS改定件数の変化率
     - `analyst_upgrade_streak`: 連続アップグレード日数
     - `target_price_gap_change`: 目標株価乖離の日次変化

4. 1ヶ月間データ蓄積後、backfill で精度比較

**ファイル変更**:
- `src/qtp/data/database.py` — スキーマ変更、日次蓄積メソッド追加
- `src/qtp/features/tier5_alternative.py` — 時系列対応
- `scripts/daily_predict.sh` — MCPデータ取得追加
- `src/qtp/data/fetchers/mcp_alternative.py` — 日次蓄積モード

**期待効果**: conf 55%+ 精度 61.3% → 63-65% (+2-4pp)
**所要時間**: 実装2時間 + 蓄積1ヶ月
**依存**: なし

---

### C3. 確信度キャリブレーション

**目的**: モデルの出力確率を実際の的中率に一致させる（65%+の過信問題を解決）

**現状の問題**:
- モデルが65%と出力しても、実際の的中率は53.7%
- 70%以上は25%しか当たらない（逆シグナル）
- ポジションサイジングが確率に基づくため、不正確な確率 → 不適切なサイズ

**実装計画**:

1. `src/qtp/models/calibration.py` を新規作成:
```python
from sklearn.calibration import CalibratedClassifierCV
from sklearn.isotonic import IsotonicRegression

class ProbabilityCalibrator:
    """Post-hoc probability calibration."""

    def __init__(self, method="isotonic"):
        # isotonic: ノンパラメトリック（少量データに強い）
        # sigmoid: Platt Scaling（パラメトリック）
        self.method = method
        self.calibrator = None

    def fit(self, raw_proba, actual_labels):
        """OOS予測確率と実際のラベルでキャリブレーションモデルを学習."""
        if self.method == "isotonic":
            self.calibrator = IsotonicRegression(out_of_bounds="clip")
            self.calibrator.fit(raw_proba, actual_labels)
        else:
            from sklearn.linear_model import LogisticRegression
            self.calibrator = LogisticRegression()
            self.calibrator.fit(raw_proba.reshape(-1, 1), actual_labels)

    def transform(self, raw_proba):
        """生の確率をキャリブレーション済み確率に変換."""
        if self.method == "isotonic":
            return self.calibrator.transform(raw_proba)
        return self.calibrator.predict_proba(raw_proba.reshape(-1, 1))[:, 1]
```

2. パイプラインへの統合:
   - Walk-Forward CV の各 fold で:
     - Train/Val split → モデル学習
     - Val 上で raw_proba を取得
     - Val の (raw_proba, actual_label) で Calibrator を学習
     - Test 上で calibrated_proba を使用
   - これにより「60%と言ったら本当に60%当たる」状態に

3. backfill で検証:
   - キャリブレーション前後の accuracy by confidence を比較
   - 期待: 各バケットで accuracy ≈ confidence になる

**ファイル変更**:
- `src/qtp/models/calibration.py` — 新規
- `src/qtp/pipeline.py` — predict 時にキャリブレーション適用
- `scripts/backfill_predictions.py` — キャリブレーション付き backfill

**期待効果**: 65%+の逆転問題が解消、全体精度は維持、ポジションサイジングが正確に
**所要時間**: 1.5時間
**依存**: なし

---

### B3. 出来高プロファイル特徴量

**目的**: 出来高の「質」を分析し、機関投資家の動向を捕捉する

**現状の問題**:
- volume_ratio_20d（20日平均比）のみ
- 上昇時の出来高と下落時の出来高の区別がない
- 出来高急増（ブレイクアウト兆候）を検出できない

**実装計画**:

1. `src/qtp/features/tier2_volatility.py` に追加（既存ファイル）:

```python
# 上昇日出来高 vs 下落日出来高
@reg.register("volume_up_down_ratio", FeatureTier.TIER2_VOLATILITY, lookback_days=22)
def volume_up_down_ratio(df):
    up_mask = df["close"] > df["close"].shift(1)
    up_vol = (df["volume"] * up_mask.cast(pl.Float64)).rolling_sum(20)
    down_vol = (df["volume"] * (~up_mask).cast(pl.Float64)).rolling_sum(20)
    return (up_vol / down_vol).alias("volume_up_down_ratio")

# 出来高ブレイクアウト（20日平均の2倍以上）
@reg.register("volume_breakout", FeatureTier.TIER2_VOLATILITY, lookback_days=22)
def volume_breakout(df):
    avg_vol = df["volume"].rolling_mean(20)
    return (df["volume"] / avg_vol).alias("volume_breakout")

# Accumulation/Distribution Line
@reg.register("ad_line_slope", FeatureTier.TIER2_VOLATILITY, lookback_days=22)
def ad_line_slope(df):
    clv = ((df["close"] - df["low"]) - (df["high"] - df["close"])) / (df["high"] - df["low"])
    ad = (clv * df["volume"]).cum_sum()
    return ad.pct_change(20).alias("ad_line_slope")

# On-Balance Volume トレンド
@reg.register("obv_slope_20d", FeatureTier.TIER2_VOLATILITY, lookback_days=25)
def obv_slope_20d(df):
    direction = pl.when(df["close"] > df["close"].shift(1)).then(1).otherwise(-1)
    obv = (direction * df["volume"]).cum_sum()
    return obv.pct_change(20).alias("obv_slope_20d")
```

2. feature_selection.py で重要度を確認
3. Top-20 に入るか検証、入れば phase3_selected.yaml を更新

**ファイル変更**:
- `src/qtp/features/tier2_volatility.py` — 4特徴量追加
- `configs/phase3_selected.yaml` — 有効なら追加

**期待効果**: +1-2pp（出来高は金利に次ぐ予測因子の可能性）
**所要時間**: 1時間
**依存**: なし

---

### A1. ユニバース拡大（セクター分散）

**目的**: テック偏重を解消し、学習データ量を増やす

**現状の問題**:
- 7銘柄全てがテック/金融の大型株
- テック暴落時に全銘柄同時下落 → 分散効果ゼロ
- 学習データが約4,300行（もっと欲しい）

**実装計画**:

1. セクター分散候補の選定:
```yaml
# configs/universe_diversified.yaml
universe:
  tickers:
    # Tech (既存)
    - "MSFT"
    - "GOOGL"
    - "NVDA"
    - "META"
    # Finance (既存)
    - "JPM"
    - "V"
    # Consumer (追加)
    - "AMZN"    # 既存
    - "COST"    # Costco — ディフェンシブ
    - "PG"      # P&G — 生活必需品
    # Healthcare (追加)
    - "LLY"     # Eli Lilly — GLP-1トレンド
    - "JNJ"     # J&J — ディフェンシブ
    # Energy (追加)
    - "XOM"     # Exxon — エネルギー
    # Industrial (追加)
    - "CAT"     # Caterpillar — 景気循環
```

2. 各候補銘柄の backfill 精度を事前チェック:
   - 既存モデルで各銘柄の予測精度を計測
   - 55%以上のみ採用

3. 段階的追加:
   - Phase 1: COST, PG, LLY を追加 (ディフェンシブ)
   - Phase 2: XOM, CAT, JNJ を追加 (景気循環)
   - 各段階で backfill → 全体精度が下がらないことを確認

**ファイル変更**:
- `configs/universe_diversified.yaml` — 新規
- `scripts/evaluate_new_tickers.py` — 新銘柄の事前評価スクリプト

**期待効果**: 学習データ2倍、セクター分散でドローダウン改善、シグナル数増加
**所要時間**: 2時間
**依存**: なし

---

### D1. ポジションサイジング最適化

**目的**: リスクを制御しながらリターンを最大化する

**現状の問題**:
- Kelly-inspired だが確信度のみ考慮
- 銘柄のボラティリティを考慮していない（NVDA と V で同じサイズ）
- ポートフォリオ全体のリスク管理なし

**実装計画**:

1. `src/qtp/backtest/signals.py` の `PositionSizer` を改修:

```python
class AdvancedPositionSizer:
    def __init__(self, max_position_pct=0.05, max_portfolio_pct=0.30,
                 target_volatility=0.15, kelly_fraction=0.5):
        self.max_position_pct = max_position_pct
        self.max_portfolio_pct = max_portfolio_pct
        self.target_volatility = target_volatility  # 年率15%目標
        self.kelly_fraction = kelly_fraction  # Half-Kelly

    def size(self, signal, ticker_volatility, current_exposure):
        # 1. Kelly Criterion (Half)
        win_rate = signal.confidence
        avg_win = 0.02  # 過去データから
        avg_loss = 0.015
        kelly = (win_rate * avg_win - (1 - win_rate) * avg_loss) / avg_win
        kelly_size = kelly * self.kelly_fraction

        # 2. Volatility targeting
        vol_scale = self.target_volatility / (ticker_volatility * (252 ** 0.5))

        # 3. Position cap
        size = min(kelly_size, vol_scale, self.max_position_pct)

        # 4. Portfolio exposure cap
        if current_exposure + size > self.max_portfolio_pct:
            size = max(0, self.max_portfolio_pct - current_exposure)

        return size
```

2. バックテストに統合:
   - backtest_walk_forward.py でボラティリティベースサイジングを適用
   - 固定サイズ vs 最適サイズの PnL 比較

**ファイル変更**:
- `src/qtp/backtest/signals.py` — AdvancedPositionSizer
- `scripts/backtest_walk_forward.py` — サイジング統合

**期待効果**: MaxDD -6.6% → -3% 以下、Sharpe 改善
**所要時間**: 1.5時間
**依存**: C3（キャリブレーション）があるとより効果的

---

### E2. 自動再学習トリガー

**目的**: 精度低下を自動検出し、モデルを再学習する

**現状の問題**:
- 市場環境の変化（レジームチェンジ）でモデルが陳腐化
- 手動で `make train` を実行する必要がある

**実装計画**:

1. `scripts/auto_retrain.sh` を作成:
```bash
#!/bin/bash
# 週次で精度チェック → 必要なら再学習

cd /Users/wakiryoutarou/quant-trading-pipeline
VENV=".venv/bin"

# 1. 直近30日の精度を取得
ACCURACY=$($VENV/python -c "
from qtp.data.database import QTPDatabase
from pathlib import Path
db = QTPDatabase(Path('data/qtp.db'))
s = db.get_accuracy_summary(days=30)
print(s.get('accuracy', 0) if s.get('total', 0) >= 10 else 'insufficient')
")

if [ "$ACCURACY" = "insufficient" ]; then
    echo "Not enough data for accuracy check"
    exit 0
fi

# 2. 閾値チェック（55%以下なら再学習）
NEEDS_RETRAIN=$(python -c "print('yes' if float('$ACCURACY') < 0.55 else 'no')")

if [ "$NEEDS_RETRAIN" = "yes" ]; then
    echo "Accuracy $ACCURACY < 55%, retraining..."
    $VENV/qtp fetch
    $VENV/qtp train -m configs/phase3_best.yaml
    echo "Retrained. New model deployed."
else
    echo "Accuracy $ACCURACY >= 55%, no retrain needed."
fi
```

2. cron に週次チェックを追加:
```
0 10 * * 6  /path/to/auto_retrain.sh  # 毎週土曜10時
```

3. Slack/メール通知（オプション）:
   - 再学習が発生したら通知
   - 精度が50%以下に落ちたらアラート

**ファイル変更**:
- `scripts/auto_retrain.sh` — 新規
- Makefile — `make auto-retrain` 追加

**期待効果**: モデル陳腐化の防止、常に最新の市場環境に適応
**所要時間**: 1時間
**依存**: 予測データの蓄積（最低30日分）

---

## Phase 4B: 中難易度（2-4時間）

### C2. メタラーナー（スタッキング）

**目的**: モデルの強みを状況に応じて動的に組み合わせる

**実装計画**:

1. `src/qtp/models/stacking.py` を新規作成:

```python
class StackingEnsemble:
    """Level-1 models + Level-2 meta-learner."""

    def __init__(self):
        self.level1_models = [
            ("lgbm", LGBMPipeline(clf_params=TUNED_LGBM)),
            ("xgb", XGBPipeline(clf_params=TUNED_XGB)),
            ("rf", RandomForestWrapper()),  # 新規追加
        ]
        self.meta_learner = LogisticRegression()

    def fit(self, X, y_dir, y_mag):
        # Level-1: 各モデルをCV予測で学習
        meta_features = []
        for name, model in self.level1_models:
            oof_preds = cross_val_predict(model, X, y_dir, cv=3)
            meta_features.append(oof_preds)

        # Level-2: メタラーナー学習
        meta_X = np.column_stack(meta_features)
        self.meta_learner.fit(meta_X, y_dir)

    def predict_proba(self, X):
        preds = [model.predict_proba(X) for _, model in self.level1_models]
        meta_X = np.column_stack(preds)
        return self.meta_learner.predict_proba(meta_X)[:, 1]
```

2. RandomForest ラッパーを追加:
   - 決定木ベースの多様性確保（LGBM/XGBと異なるバイアス）

3. backfill で比較: スタッキング vs 単純加重平均

**ファイル変更**:
- `src/qtp/models/stacking.py` — 新規
- `src/qtp/models/random_forest.py` — 新規
- `scripts/evaluate_stacking.py` — 評価スクリプト

**期待効果**: +1-3pp
**所要時間**: 3時間
**依存**: なし

---

### B1. クロスセクション特徴量

**目的**: 銘柄間の相対的な強弱を捕捉する

**実装計画**:

1. `src/qtp/features/cross_sectional.py` を新規作成:

```python
# FeatureEngine.build_multi_ticker_dataset 内で計算
# 各日付で全銘柄の特徴量を比較

def compute_cross_sectional_features(dataset):
    """銘柄間の相対比較特徴量を追加."""

    # 1. 相対リターン: 自分のリターン - 全銘柄平均リターン
    # → 市場全体が上がっている中での相対的な強さ
    dataset = dataset.with_columns([
        (pl.col("ret_21d") - pl.col("ret_21d").mean().over("date"))
        .alias("relative_strength_21d"),
    ])

    # 2. セクター内順位 (0-1): その日で何番目に強いか
    dataset = dataset.with_columns([
        pl.col("ret_21d").rank().over("date")
        / pl.col("ret_21d").count().over("date")
        .alias("rank_momentum"),
    ])

    # 3. 銘柄間相関レジーム: 全銘柄の相関が高い→リスクオフ
    # rolling 63日の銘柄間相関を計算

    # 4. ベータ: SPY に対するベータ（感応度）

    return dataset
```

2. `FeatureEngine.build_multi_ticker_dataset` に統合:
   - 全銘柄のデータが揃った後にクロスセクション特徴量を計算
   - 計算順序: 個別特徴量 → 結合 → クロスセクション特徴量

**ファイル変更**:
- `src/qtp/features/cross_sectional.py` — 新規
- `src/qtp/features/engine.py` — build_multi_ticker_dataset に統合

**期待効果**: +2-4pp（特にロングショート戦略との組み合わせで大）
**所要時間**: 2.5時間
**依存**: A1（銘柄が多いほど効果的）

---

### E1. Streamlit ダッシュボード

**目的**: 精度・PnL・シグナルをリアルタイムに可視化

**実装計画**:

1. `dashboard/app.py` を新規作成:

```python
import streamlit as st
from qtp.data.database import QTPDatabase

st.title("QTP Trading Dashboard")

# タブ: Overview / Predictions / Accuracy / Backtest
tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Predictions", "Accuracy", "Backtest"])

with tab1:
    # 精度推移グラフ（30日移動平均）
    # 今日のシグナル一覧
    # ポートフォリオエクスポージャー
    pass

with tab2:
    # 直近の予測一覧（OK/NG 表示）
    # フィルタ: 銘柄、確信度、日付範囲
    pass

with tab3:
    # 確信度キャリブレーションカーブ
    # 銘柄別精度ヒートマップ
    # 月次精度推移
    pass

with tab4:
    # エクイティカーブ
    # ドローダウン推移
    # 戦略比較チャート
    pass
```

2. pyproject.toml に streamlit 依存追加
3. Makefile: `make dashboard` → `streamlit run dashboard/app.py`

**ファイル変更**:
- `dashboard/app.py` — 新規
- `dashboard/pages/` — 各ページ
- `pyproject.toml` — streamlit 依存
- `Makefile` — dashboard コマンド

**期待効果**: リアルタイムモニタリング、直感的な精度把握
**所要時間**: 3時間
**依存**: 蓄積データが多いほど有用

---

### D2. ストップロス・利確ルール

**目的**: 個別トレードのリスク管理を強化

**実装計画**:

1. `src/qtp/backtest/risk_management.py` を新規作成:

```python
class TradeManager:
    def __init__(self, stop_loss_pct=-0.02, take_profit_pct=0.05,
                 trailing_stop_pct=0.03, max_hold_days=10):
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.trailing_stop_pct = trailing_stop_pct
        self.max_hold_days = max_hold_days

    def check_exit(self, entry_price, current_price, peak_price, days_held):
        """エグジット判定. 理由を返す."""
        pnl = (current_price - entry_price) / entry_price

        # ストップロス
        if pnl <= self.stop_loss_pct:
            return "stop_loss"

        # 利確
        if pnl >= self.take_profit_pct:
            return "take_profit"

        # トレーリングストップ
        peak_pnl = (peak_price - entry_price) / entry_price
        drawdown_from_peak = (current_price - peak_price) / peak_price
        if peak_pnl > 0.02 and drawdown_from_peak < -self.trailing_stop_pct:
            return "trailing_stop"

        # 最大保有期間
        if days_held >= self.max_hold_days:
            return "max_hold"

        return None  # Hold
```

2. 日次バックテストに統合:
   - 現在は10日間丸ごとホールド
   - 日次でストップロス/利確チェックを挟む
   - OHLCV の日次データを使って intraday high/low も考慮

3. ストップロス有無の PnL 比較

**ファイル変更**:
- `src/qtp/backtest/risk_management.py` — 新規
- `scripts/backtest_walk_forward.py` — 日次リスク管理統合

**期待効果**: MaxDD -6.6% → -3%、大損トレードの制限
**所要時間**: 2.5時間
**依存**: D1（ポジションサイジングと組み合わせるとより効果的）

---

## Phase 4C: 高難易度（4時間以上）

### C1. 時系列特化モデル（LSTM / Transformer）

**目的**: 過去N日の「順序」情報を使って予測する

**実装計画**:

1. `src/qtp/models/lstm.py` を新規作成:

```python
import torch
import torch.nn as nn

class LSTMPredictor(nn.Module):
    def __init__(self, n_features, hidden_size=64, n_layers=2, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden_size, n_layers,
                           batch_first=True, dropout=dropout)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # x: (batch, seq_len, n_features)
        lstm_out, _ = self.lstm(x)
        last_hidden = lstm_out[:, -1, :]
        return self.classifier(last_hidden).squeeze(-1)
```

2. データローダー:
   - 過去20日分の特徴量を窓として切り出し
   - (batch, 20, n_features) のテンソルに変換

3. アンサンブル統合:
   - LSTM の出力確率を Level-1 モデルとしてスタッキングに追加
   - LGBM + XGB + LSTM の3モデルアンサンブル

**追加依存**: torch, PyTorch
**期待効果**: +2-5pp（時系列パターン学習による）
**所要時間**: 6時間
**依存**: なし（ただし GPU があると学習が速い）

---

### B2. ニュースセンチメント特徴量

**目的**: テキスト情報から市場センチメントを数値化

**実装計画**:

1. 日次ニュース取得:
   - `/market-news` スキルまたは WebSearch で各銘柄のニュースを取得
   - Claude API でセンチメントスコア化（-1〜+1）

2. SQLite に蓄積:
```sql
CREATE TABLE news_sentiment (
    ticker TEXT,
    date TEXT,
    headline TEXT,
    sentiment_score REAL,  -- -1 to +1
    source TEXT,
    fetched_at TIMESTAMP
);
```

3. 特徴量化:
   - `sentiment_avg_3d`: 過去3日間の平均センチメント
   - `sentiment_change`: センチメントの変化率
   - `news_volume`: ニュース件数（注目度の代理変数）

**追加依存**: Claude API（コスト: ~$0.01/銘柄/日）
**期待効果**: +2-4pp（イベントドリブンな動きの予測）
**所要時間**: 5時間
**依存**: A3（日次蓄積の仕組み）

---

### A2. 日本株展開

**実装計画**:

1. dexter-jp MCP で日本株データ取得
2. 日本市場カレンダー対応（exchange-calendars ライブラリ）
3. 特徴量: 同じ Tier1-4 を使用（テクニカル指標は市場非依存）
4. 追加特徴量:
   - 為替（USD/JPY）影響
   - 日銀政策金利
   - 信用残（日本市場特有）
5. 別モデルで学習（米国モデルの転移学習は効果薄い可能性）

**期待効果**: 取引機会の2倍化、時間帯分散
**所要時間**: 8時間
**依存**: A1（パイプラインが複数市場対応になっていること）

---

### D3. 実取引API連携（Alpaca）

**実装計画**:

1. `src/qtp/execution/alpaca_client.py`:
```python
import alpaca_trade_api as tradeapi

class AlpacaExecutor:
    def __init__(self, api_key, secret_key, paper=True):
        base_url = "https://paper-api.alpaca.markets" if paper else "https://api.alpaca.markets"
        self.api = tradeapi.REST(api_key, secret_key, base_url)

    def execute_signals(self, signals):
        for signal in signals:
            if signal.direction == 1 and signal.confidence >= 0.55:
                size = self.position_sizer.size(signal)
                self.api.submit_order(
                    symbol=signal.ticker,
                    qty=int(size / current_price),
                    side='buy',
                    type='limit',
                    time_in_force='day',
                    limit_price=current_price * 1.001,
                )
```

2. ペーパートレード期間: 最低1ヶ月
3. リスク管理:
   - 1銘柄最大5%
   - ポートフォリオ最大30%
   - 日次PnLリミット -2%

**追加依存**: alpaca-trade-api
**期待効果**: 完全自動取引
**所要時間**: 8時間
**依存**: D1, D2, C3（リスク管理が先）

---

## 実行ロードマップ

### Week 1: 基盤強化
| 日 | タスク | エージェント並列 |
|----|--------|----------------|
| Day 1 | C3 キャリブレーション + B3 出来高特徴量 | 2並列 |
| Day 2 | A3 Tier5日次蓄積 + D1 ポジションサイジング | 2並列 |
| Day 3 | A1 ユニバース拡大 + E2 自動再学習 | 2並列 |

### Week 2: モデル改善
| 日 | タスク | エージェント並列 |
|----|--------|----------------|
| Day 4 | C2 スタッキング + B1 クロスセクション | 2並列 |
| Day 5 | E1 Streamlit ダッシュボード + D2 ストップロス | 2並列 |

### Week 3: 高度な改善
| 日 | タスク |
|----|--------|
| Day 6-7 | C1 LSTM モデル |
| Day 8 | B2 ニュースセンチメント |

### Week 4: 実取引準備
| 日 | タスク |
|----|--------|
| Day 9-10 | D3 Alpaca ペーパートレード |
| Day 11 | A2 日本株展開（開始） |

---

## 精度目標

| マイルストーン | conf 55%+ 精度 | 根拠 |
|---------------|---------------|------|
| 現在 | 61.3% | ベースライン |
| Week 1 完了 | 64-66% | キャリブレーション + 出来高 + サイジング |
| Week 2 完了 | 66-68% | スタッキング + クロスセクション |
| Week 3 完了 | 68-72% | LSTM + センチメント |
| Tier5 蓄積1ヶ月後 | 70-75% | 時系列Alternative Dataの効果 |
