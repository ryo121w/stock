# QTP 統合パイプライン改修計画

## 問題の定義

現状: QTP, MAGI, TradingView, 掲示板がバラバラに動き、人間が「なんとなく」統合している
結果: 判断がブレる（MSFTがBUY→WATCH→BUY→WATCHと4回変わった）

目標: **7ステップの自動ゲートシステム**で、全ゲート通過銘柄だけが最終推奨に残る

---

## アーキテクチャ

```
入力: ユニバース全銘柄（現在14銘柄）
  │
  ├─ Gate 1: QTP定量スコア ─── 足切り60%未満はOUT
  │
  ├─ Gate 2: テクニカル確認 ─── RSI>75 or MACD下向き はOUT
  │
  ├─ Gate 3: ファンダチェック ── 利益減益 or 目標株価<現値 はOUT
  │
  ├─ Gate 4: MAGI定性レビュー ─ 否決(2-1 AVOID以上)はOUT
  │
  ├─ Gate 5: センチメント確認 ── 極端な楽観はOUT（逆指標）
  │
  ├─ Gate 6: 統合スコア算出 ─── 重みは過去データで最適化済み
  │
  └─ Gate 7: 最終判定 ──────── BUY / WATCH / AVOID + ポジションサイズ
  │
出力: 最終推奨リスト（通常1-3銘柄）+ 理由 + エントリー条件
```

---

## Gate 1: QTP 定量スコア

### 目的
機械学習モデルの予測で足切り。「モデルが自信を持てない銘柄」を排除。

### 実装

```python
# src/qtp/gates/gate1_qtp.py

class Gate1_QTP:
    """定量モデルによる足切りゲート"""

    PASS_THRESHOLD = 0.55  # conf 55%以上で通過

    def evaluate(self, ticker: str) -> GateResult:
        """
        1. QTP predict を実行
        2. confidence >= 55% かつ direction == UP なら通過
        3. 歴史的精度も確認（backfill結果から）
        """
        # 今日の予測
        prediction = self.get_latest_prediction(ticker)

        # 歴史的精度（backfillから）
        historical_accuracy = self.get_historical_accuracy(ticker)

        # 通過条件
        passed = (
            prediction.confidence >= self.PASS_THRESHOLD
            and prediction.direction == 1  # UP
            and historical_accuracy >= 0.53  # 歴史的にコイン投げ以上
        )

        return GateResult(
            gate="QTP",
            passed=passed,
            score=prediction.confidence * 100,
            reason=f"conf={prediction.confidence:.1%}, hist_acc={historical_accuracy:.1%}",
        )
```

### 通過条件
- conf >= 55% かつ direction == UP
- 歴史的精度 >= 53%（backfillデータ）
- **両方満たさないと通過しない**（MSFT問題の再発防止: conf 84.5% でも hist_acc 51% なら不通過）

### テスト方法
- backfillデータで「Gate1通過銘柄の的中率」vs「全銘柄の的中率」を比較
- Gate1 通過後の的中率が 60%+ であることを確認

---

## Gate 2: テクニカル確認

### 目的
過熱銘柄・下降トレンド銘柄を排除。TradingView MCP から取得。

### 実装

```python
# src/qtp/gates/gate2_technical.py

class Gate2_Technical:
    """テクニカル指標による確認ゲート"""

    def evaluate(self, ticker: str) -> GateResult:
        """
        TradingView MCP or YFinance から取得:
        1. RSI(14): 75以上は過熱→OUT, 25以下は売られすぎ→加点
        2. MACD: ヒストグラムが上向きか（モメンタム改善中か）
        3. SMA200: 株価がSMA200の上か下か（トレンド）
        4. 出来高: 20日平均比で異常値がないか
        """
        rsi = self.get_rsi(ticker)
        macd_improving = self.is_macd_improving(ticker)
        above_sma200 = self.is_above_sma200(ticker)

        # 絶対的排除条件
        if rsi > 75:
            return GateResult(passed=False, reason=f"RSI {rsi:.0f} 過熱")

        # スコア算出（0-100）
        score = 50
        score += (50 - rsi) * 0.5  # RSI低いほど加点（max +25）
        score += 15 if macd_improving else -10
        score += 10 if above_sma200 else -15
        score = max(0, min(100, score))

        passed = score >= 40  # 40点以上で通過

        return GateResult(
            gate="Technical",
            passed=passed,
            score=score,
            reason=f"RSI={rsi:.0f}, MACD={'↑' if macd_improving else '↓'}, SMA200={'上' if above_sma200 else '下'}",
        )
```

### 通過条件
- RSI < 75（過熱でない）
- テクニカルスコア >= 40
- **RSI > 75 は無条件 OUT**（JNJ問題の再発防止: 52週92%位置でアップサイドなし）

### データソース
- 優先: TradingView デスクトップMCP（リアルタイム）
- フォールバック: YFinance + QTP特徴量計算

---

## Gate 3: ファンダメンタルチェック

### 目的
業績悪化銘柄を排除。MCP ツールから取得。

### 実装

```python
# src/qtp/gates/gate3_fundamental.py

class Gate3_Fundamental:
    """ファンダメンタルズ確認ゲート"""

    def evaluate(self, ticker: str) -> GateResult:
        """
        stock-tools MCP から取得:
        1. 売上成長率: マイナスは警告
        2. 利益成長率: マイナスはOUT
        3. EPS改定トレンド: DOWNGRADE はOUT
        4. アナリスト目標株価: 現値以下はOUT（日本郵船問題の防止）
        5. ROE: 10%未満は警告
        """
        quote = self.fetch_yahoo_quote(ticker)
        earnings_trend = self.fetch_earnings_trend(ticker)
        estimates = self.fetch_analyst_estimates(ticker)

        # 絶対的排除条件
        if earnings_trend.signal == "DOWNGRADE":
            return GateResult(passed=False, reason="EPS DOWNGRADE")

        if estimates.target_price and estimates.target_price < quote.price:
            return GateResult(passed=False, reason=f"目標{estimates.target_price} < 現値{quote.price}")

        # スコア算出
        score = 50
        score += min(20, quote.revenue_growth * 0.5)  # 成長加点
        score += min(15, quote.earnings_growth * 0.3)
        score += 10 if earnings_trend.signal == "UPGRADE" else 0
        score += 5 if quote.roe > 15 else -5
        score = max(0, min(100, score))

        passed = score >= 35

        return GateResult(
            gate="Fundamental",
            passed=passed,
            score=score,
            reason=f"売上{quote.revenue_growth:+.1f}%, EPS改定={earnings_trend.signal}",
        )
```

### 通過条件
- EPS改定が DOWNGRADE でない
- アナリスト目標株価 > 現在株価
- ファンダスコア >= 35
- **目標株価 < 現値 は無条件 OUT**（日本郵船問題の防止）

---

## Gate 4: MAGI 定性レビュー

### 目的
数値では見えないリスクを3つの視点で評価。

### 実装

```python
# src/qtp/gates/gate4_magi.py

class Gate4_MAGI:
    """MAGI 3体レビューゲート"""

    def evaluate(self, ticker: str, research_data: dict) -> GateResult:
        """
        3エージェントを並列起動:
        - MELCHIOR: 数値のみ → BUY/HOLD/AVOID
        - BALTHASAR: リスクのみ → BUY/HOLD/AVOID + 推奨配分
        - CASPER: センチメントのみ → BUY/HOLD/AVOID + タイミング

        投票集計:
        - 3-0 BUY: score=95, 通過
        - 2-1 BUY: score=75, 通過
        - 1-1-1: score=50, 通過（ただし配分半減）
        - 2-1 AVOID: score=25, 不通過
        - 3-0 AVOID: score=5, 不通過（エントリー禁止）
        """
        votes = self.run_magi_parallel(ticker, research_data)

        buy_count = sum(1 for v in votes if v == "BUY")
        avoid_count = sum(1 for v in votes if v == "AVOID")

        if buy_count >= 2:
            score = 70 + buy_count * 10  # 2-1:80, 3-0:100 → 調整後 75, 95
            passed = True
        elif buy_count == 1 and avoid_count <= 1:
            score = 50
            passed = True  # 分裂: 通過だが配分半減
        else:
            score = 25 - avoid_count * 10
            passed = False

        return GateResult(
            gate="MAGI",
            passed=passed,
            score=score,
            dissent=self.get_dissent_reason(votes),
        )
```

### 通過条件
- BUY票 >= 2（過半数）
- 3-0 AVOID は**永久不通過**（次の評価でリセットされるまで）
- **判断は1回限り。新情報がない限り変更しない**（ブレ防止ルール）

### ブレ防止メカニズム

```python
# MAGI結果をSQLiteに保存。同じ銘柄を再評価する条件:
# 1. 前回評価から14日以上経過
# 2. 決算発表があった
# 3. 重大ニュース（M&A、FDA等）があった
# 上記以外は前回の評価結果をキャッシュから返す
```

---

## Gate 5: センチメント確認

### 目的
「全員が楽観 = 天井」の逆指標を検出。

### 実装

```python
# src/qtp/gates/gate5_sentiment.py

class Gate5_Sentiment:
    """市場センチメント確認ゲート"""

    def evaluate(self, ticker: str) -> GateResult:
        """
        確認項目:
        1. Yahoo掲示板: 極端な楽観（全員が買い推奨）→ 警告
        2. アナリスト全員買い: 10人中10人BUY → 逆指標警告
        3. VIX / 日経VI: 極端な低ボラ → 油断警告
        4. 信用残: 買い残が極端に多い → 需給悪化警告

        注意: これは「排除」ではなく「警告」ゲート。
        通過はするが、配分を減らす。
        """
        warnings = []
        score = 70  # デフォルトは通過

        # アナリスト全員買い = 逆指標
        if self.analyst_all_buy(ticker):
            warnings.append("アナリスト全員買い（逆指標リスク）")
            score -= 15

        # 掲示板が極端に楽観
        if self.board_euphoric(ticker):
            warnings.append("掲示板が極端に楽観的")
            score -= 20

        # 逆に、掲示板が悲観的 = 逆張りチャンス
        if self.board_pessimistic(ticker):
            score += 10  # 加点

        return GateResult(
            gate="Sentiment",
            passed=True,  # このゲートは常に通過（配分調整のみ）
            score=max(0, min(100, score)),
            warnings=warnings,
        )
```

### 特殊ルール
- **排除はしない**（通過率100%）
- 代わりに**配分を調整**する
- 極端な楽観 → 配分を半減
- 極端な悲観 → 配分を増加（逆張り）

---

## Gate 6: 統合スコア算出

### 目的
全ゲートのスコアを最適な重みで統合。

### 実装

```python
# src/qtp/gates/gate6_integration.py

class Gate6_Integration:
    """統合スコア算出"""

    # デフォルト重み（後でbackfillデータで最適化）
    DEFAULT_WEIGHTS = {
        "qtp": 0.25,
        "technical": 0.15,
        "fundamental": 0.25,
        "magi": 0.25,
        "sentiment": 0.10,
    }

    def calculate(self, gate_results: dict[str, GateResult]) -> float:
        """
        統合スコア = Σ(gate_score × weight)

        重みの最適化:
        - backfillデータで各ゲートスコアと実際のリターンの相関を計算
        - 相関が高いゲートの重みを増やす
        - 定期的に再最適化（月1回）
        """
        weights = self.load_optimized_weights()  # SQLiteから

        total = 0
        for gate_name, result in gate_results.items():
            if gate_name in weights:
                total += result.score * weights[gate_name]

        return total

    def optimize_weights(self):
        """
        過去のbackfillデータで最適重みを算出:
        1. 各ゲートスコアと実際のN日後リターンの相関を計算
        2. 相関係数を正規化して重みに変換
        3. SQLiteに保存
        """
        # 実装: scipy.optimize.minimize で最適化
        pass
```

### 重み最適化の方法

```
1. backfill データから各銘柄×日付の:
   - Gate1スコア, Gate2スコア, ..., Gate5スコア
   - 実際の10日後リターン
   を取得

2. scipy.optimize.minimize で:
   目的関数 = -1 × (統合スコアと実際リターンの相関係数)
   制約: 全重みの合計 = 1.0, 各重み >= 0.05

3. 最適重みをSQLiteに保存
4. 月1回再最適化
```

---

## Gate 7: 最終判定

### 目的
統合スコアから判定 + ポジションサイズ + エントリー条件を出力。

### 実装

```python
# src/qtp/gates/gate7_verdict.py

class Gate7_Verdict:
    """最終判定"""

    THRESHOLDS = {
        "STRONG_BUY": 80,
        "BUY": 65,
        "WATCH": 50,
        "HOLD": 35,
        "AVOID": 0,
    }

    def judge(self, integrated_score: float, gate_results: dict) -> FinalVerdict:
        """
        1. 統合スコアから判定
        2. BALTHASARの推奨配分を採用
        3. センチメント警告があれば配分調整
        4. エントリー条件を設定
        5. 損切り・利確ラインを設定
        """
        # 判定
        verdict = "AVOID"
        for label, threshold in self.THRESHOLDS.items():
            if integrated_score >= threshold:
                verdict = label
                break

        # 配分（BALTHASAR推奨 × センチメント調整）
        base_allocation = gate_results["magi"].balthasar_allocation
        sentiment_factor = gate_results["sentiment"].score / 70  # 70が基準
        allocation = base_allocation * sentiment_factor

        # エントリー条件
        entry = self.calculate_entry(gate_results)
        stop_loss = entry.price * 0.85  # -15%
        target = gate_results["fundamental"].target_price

        return FinalVerdict(
            verdict=verdict,
            score=integrated_score,
            allocation=allocation,
            entry=entry,
            stop_loss=stop_loss,
            target=target,
            # ブレ防止: この判定をSQLiteに保存
            # 新情報がない限り14日間変更しない
        )
```

### ブレ防止ルール（最重要）

```python
class VerdictCache:
    """一度出した判定は簡単に変えない"""

    LOCK_DAYS = 14  # 14日間は判定をロック

    def get_or_evaluate(self, ticker: str) -> FinalVerdict:
        cached = self.db.get_cached_verdict(ticker)

        if cached and not self.should_re_evaluate(cached):
            return cached  # キャッシュを返す（ブレない）

        # 再評価が必要な条件
        return self.full_evaluation(ticker)

    def should_re_evaluate(self, cached) -> bool:
        """再評価すべき条件（これ以外は前回の判定を維持）"""
        days_since = (date.today() - cached.date).days

        return (
            days_since >= self.LOCK_DAYS           # 14日経過
            or self.had_earnings(cached.ticker)     # 決算があった
            or self.had_major_news(cached.ticker)   # 重大ニュース
            or self.price_moved_15pct(cached)        # 株価が15%以上動いた
        )
```

---

## SQLite スキーマ追加

```sql
-- Gate評価結果の保存
CREATE TABLE IF NOT EXISTS gate_evaluations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker TEXT NOT NULL,
    evaluation_date TEXT NOT NULL,
    gate1_qtp_score REAL,
    gate1_passed INTEGER,
    gate2_technical_score REAL,
    gate2_passed INTEGER,
    gate3_fundamental_score REAL,
    gate3_passed INTEGER,
    gate4_magi_score REAL,
    gate4_passed INTEGER,
    gate4_votes TEXT,  -- JSON: {"melchior":"BUY","balthasar":"HOLD","casper":"BUY"}
    gate5_sentiment_score REAL,
    gate5_warnings TEXT,  -- JSON
    integrated_score REAL,
    final_verdict TEXT,  -- STRONG_BUY/BUY/WATCH/HOLD/AVOID
    allocation REAL,
    entry_price REAL,
    stop_loss REAL,
    target_price REAL,
    locked_until TEXT,  -- 判定ロック期限
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(ticker, evaluation_date)
);

-- 重み最適化の履歴
CREATE TABLE IF NOT EXISTS weight_optimization (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    weights JSON NOT NULL,  -- {"qtp":0.25,"technical":0.15,...}
    correlation REAL,  -- 最適化時の相関係数
    n_samples INTEGER,
    optimized_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

---

## 実装ロードマップ

### Phase 1: 基盤（2時間）
| タスク | ファイル | 内容 |
|--------|---------|------|
| GateResult データクラス | src/qtp/gates/__init__.py | 共通型定義 |
| Gate1 QTP | src/qtp/gates/gate1_qtp.py | 定量スコア |
| Gate2 Technical | src/qtp/gates/gate2_technical.py | テクニカル |
| Gate3 Fundamental | src/qtp/gates/gate3_fundamental.py | ファンダ |
| SQLiteスキーマ | src/qtp/data/database.py | テーブル追加 |

### Phase 2: MAGI + センチメント（2時間）
| タスク | ファイル | 内容 |
|--------|---------|------|
| Gate4 MAGI | src/qtp/gates/gate4_magi.py | 3体レビュー |
| Gate5 Sentiment | src/qtp/gates/gate5_sentiment.py | センチメント |
| VerdictCache | src/qtp/gates/verdict_cache.py | ブレ防止 |

### Phase 3: 統合 + 最適化（2時間）
| タスク | ファイル | 内容 |
|--------|---------|------|
| Gate6 Integration | src/qtp/gates/gate6_integration.py | 重み統合 |
| Gate7 Verdict | src/qtp/gates/gate7_verdict.py | 最終判定 |
| 重み最適化 | scripts/optimize_weights.py | backfillデータで最適化 |
| パイプラインオーケストレーター | src/qtp/gates/pipeline.py | 7ゲート一気通貫 |

### Phase 4: スキル + テスト（2時間）
| タスク | ファイル | 内容 |
|--------|---------|------|
| /qtp-signal 改修 | ~/.claude/skills/qtp-signal/SKILL.md | 7ゲート対応 |
| CLI コマンド | src/qtp/cli.py | qtp signal <ticker> |
| テスト | tests/test_gates/ | 各ゲートのユニットテスト |
| backfill検証 | scripts/validate_gates.py | 過去データで精度検証 |

### Phase 5: 検証 + 運用（1時間）
| タスク | 内容 |
|--------|------|
| 三井住友FGで全ゲート通し確認 | 本番データで7ゲート実行 |
| 全14銘柄でバッチ実行 | 結果を確認・調整 |
| daily_predict.sh 統合 | 毎朝のパイプラインに組み込み |

---

## 検証方法

### backfill で統合パイプラインを事後検証

```python
# scripts/validate_gates.py

"""
過去データで7ゲートパイプラインを走らせ、
「全ゲート通過銘柄のリターン」vs「一部不通過銘柄のリターン」を比較

期待結果:
- 全ゲート通過: 平均リターン +3% 以上 (10日)
- 一部不通過: 平均リターン +0.5% 以下
- 差が統計的に有意（t検定 p < 0.05）
"""
```

### 三井住友FG の事後検証（30日後）

```
2026年5月上旬に検証:
- エントリー ¥5,350 → 30日後の株価は？
- 7ゲート全通過の判断は正しかったか？
- 日銀利上げは実現したか？
- 結果をフィードバックして重み再最適化
```

---

## 成功基準

| 指標 | 現状 | 目標 |
|------|------|------|
| 判断のブレ | 1銘柄で4回変更 | **0回**（ロック機能） |
| 全ゲート通過銘柄の的中率 | 未計測 | **65%+** |
| 全ゲート通過のリターン | 未計測 | **+3%+ (10日)** |
| 評価にかかる時間 | 30分+（手動） | **5分以内（自動）** |
| 重み付けの根拠 | 「なんとなく」 | **データ最適化** |
