# 設計書: オルタナティブデータ取得 v2

## 1. 背景と問題

### 現状の問題

| 問題 | 影響 | 深刻度 |
|------|------|--------|
| Tier5 特徴量が**全てゼロ重要度** | 18特徴量中11がモデルに寄与しない | 高 |
| `mcp_client.py` が**存在しない** | MCPAlternativeFetcher が ImportError | 致命的 |
| alternative_data_daily に**1日分のみ** | 7日トレンド特徴量が常に0.0 | 高 |
| Finviz/OpenInsider の**DOM スクレイピング** | DOM変更で全壊 | 高 |
| ニュースセンチメントが**キーワード採点** | 42個の固定キーワードで精度低い | 中 |
| 過去のオルタナティブデータが**存在しない** | 学習データ2年分で Tier5 は全行0.0 | 高 |

### 根本原因

1. **MCP ツールは「今」のデータしか返さない** — 過去の時系列がない
2. **日次蓄積の仕組みがない** — cron/scheduler が未実装
3. **スクレイピングに依存** — API ファーストでない
4. **mcp_client.py が未実装** — MCPAlternativeFetcher が動かない

---

## 2. 設計方針

### 原則

1. **API ファースト**: スクレイピングは最終手段。公式APIを優先
2. **無料優先**: $0 で最大効果。有料は FMP $19/月のみオプション
3. **日次蓄積**: 今日から蓄積開始 → 30日後に Tier5 が学習に寄与
4. **段階的移行**: 既存コードを壊さず、fetcher を差し替え可能に
5. **ポイントインタイム厳守**: 学習時のリーケージ防止

### データソース選定

| データ種別 | 現在 | 新ソース | 理由 |
|-----------|------|---------|------|
| インサイダー取引 | OpenInsider (scraping) | **EdgarTools (SEC EDGAR)** | 無料、全履歴、公式ソース |
| アナリスト格付け | Yahoo Finance (3ヶ月) | **Finnhub** (60回/分無料) | 豊富な履歴、安定API |
| EPS改定 | Yahoo Finance (snapshot) | **Finnhub** + 日次蓄積 | 改定トラッキング可能 |
| ニュースセンチメント | キーワード採点 (42語) | **Finnhub** Company News | 構造化ニュース |
| マーケットレジーム | yfinance (VIX/SPY) | 現状維持 + **fear-greed** lib | VIXは十分。F&G追加 |
| 目標株価 | Finviz (scraping) | **Finnhub** | スクレイピング廃止 |

---

## 3. アーキテクチャ

### 3.1 レイヤー構成

```
┌──────────────────────────────────────────────┐
│              Feature Layer (Tier5)            │
│  eps_revision, analyst_actions, insider, ...  │
├──────────────────────────────────────────────┤
│           Storage Layer (SQLite)              │
│  alternative_data (snapshot)                  │
│  alternative_data_daily (time-series)         │
├──────────────────────────────────────────────┤
│         Fetcher Layer (NEW: v2)               │
│  ┌──────────┐ ┌──────────┐ ┌──────────────┐ │
│  │ Finnhub  │ │ Edgar    │ │ FearGreed    │ │
│  │ Fetcher  │ │ Fetcher  │ │ Fetcher      │ │
│  └──────────┘ └──────────┘ └──────────────┘ │
│  ┌──────────┐ ┌──────────┐                   │
│  │ yfinance │ │ FMP      │ (optional)        │
│  │ Fetcher  │ │ Fetcher  │                   │
│  └──────────┘ └──────────┘                   │
├──────────────────────────────────────────────┤
│         Scheduler Layer (NEW)                 │
│  Daily Accumulator (cron / Claude schedule)   │
└──────────────────────────────────────────────┘
```

### 3.2 Fetcher インターフェース

```python
# src/qtp/data/fetchers/alt_base.py

class AltDataFetcher(ABC):
    """オルタナティブデータ取得の基底クラス"""

    @abstractmethod
    def fetch(self, ticker: str, **kwargs) -> dict:
        """単一銘柄のデータを取得して dict で返す"""

    @property
    @abstractmethod
    def tool_name(self) -> str:
        """DB の tool カラムに格納する名前"""

    @property
    def max_age_hours(self) -> int:
        """キャッシュの最大有効時間"""
        return 24
```

### 3.3 新規 Fetcher

#### A. FinnhubFetcher

```python
# src/qtp/data/fetchers/finnhub_.py

class FinnhubFetcher:
    """Finnhub API 統合フェッチャー (60回/分無料)"""

    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or os.environ.get("FINNHUB_API_KEY", "")
        self.client = finnhub.Client(api_key=self.api_key)
        self._rate_limiter = RateLimiter(max_calls=55, period=60)  # 余裕を持って55/分

    def fetch_eps_estimates(self, ticker: str) -> dict:
        """EPS コンセンサス予想"""
        data = self.client.company_eps_estimates(ticker, freq="quarterly")
        return {"estimates": data, "fetched_at": now_iso()}

    def fetch_recommendation_trends(self, ticker: str) -> dict:
        """アナリスト推奨トレンド (Buy/Hold/Sell/StrongBuy/StrongSell)"""
        data = self.client.recommendation_trends(ticker)
        return {"trends": data, "fetched_at": now_iso()}

    def fetch_price_target(self, ticker: str) -> dict:
        """アナリスト目標株価コンセンサス"""
        data = self.client.price_target(ticker)
        return {"target_high": data.targetHigh, "target_low": data.targetLow,
                "target_mean": data.targetMean, "target_median": data.targetMedian}

    def fetch_upgrade_downgrade(self, ticker: str) -> dict:
        """格上げ/格下げ履歴"""
        data = self.client.upgrade_downgrade(symbol=ticker)
        return {"actions": data, "fetched_at": now_iso()}

    def fetch_company_news(self, ticker: str, days_back: int = 7) -> dict:
        """企業ニュース (センチメント用)"""
        from_date = (date.today() - timedelta(days=days_back)).isoformat()
        to_date = date.today().isoformat()
        articles = self.client.company_news(ticker, _from=from_date, to=to_date)
        # Finnhub returns: headline, summary, source, url, datetime, category
        return {"articles": articles[:50], "count": len(articles)}
```

#### B. EdgarFetcher

```python
# src/qtp/data/fetchers/edgar_.py

class EdgarFetcher:
    """SEC EDGAR via EdgarTools (完全無料、全履歴)"""

    def fetch_insider_transactions(self, ticker: str, months: int = 6) -> dict:
        """インサイダー取引 (SEC Form 4)"""
        from edgar import Company
        company = Company(ticker)
        filings = company.get_filings(form="4").latest(20)  # 直近20件

        transactions = []
        for filing in filings:
            form4 = filing.obj()  # Parse Form 4
            for txn in form4.transactions:
                transactions.append({
                    "date": txn.date.isoformat(),
                    "owner": txn.owner_name,
                    "title": txn.owner_title,
                    "type": "BUY" if txn.acquired else "SELL",
                    "shares": txn.shares,
                    "price": txn.price_per_share,
                    "value": txn.shares * txn.price_per_share if txn.price_per_share else 0,
                })
        return {"transactions": transactions, "count": len(transactions)}

    def fetch_insider_history(self, ticker: str, years: int = 3) -> list[dict]:
        """過去数年のインサイダー取引履歴 (ML学習用)"""
        # EdgarTools で過去のForm 4を全取得
        # 月ごとに集約: net_buys, net_sells, net_signal
        ...
```

#### C. FearGreedFetcher

```python
# src/qtp/data/fetchers/fear_greed_.py

class FearGreedFetcher:
    """CNN Fear & Greed Index (pip install fear-greed)"""

    def fetch(self) -> dict:
        import feargreed
        data = feargreed.get()
        return {
            "score": data.score,           # 0-100
            "rating": data.description,    # "Extreme Fear" ~ "Extreme Greed"
            "previous_close": data.previous_close,
            "one_week_ago": data.one_week_ago,
            "one_month_ago": data.one_month_ago,
            "one_year_ago": data.one_year_ago,
        }
```

---

## 4. 日次蓄積パイプライン

### 4.1 実行フロー

```
毎日 09:00 JST (US プレマーケット前)
│
├─ Phase 1: マーケットレベルデータ (1回)
│   ├─ FearGreedFetcher.fetch() → alternative_data_daily(_market, fear_greed)
│   └─ yfinance VIX/SPY/TNX → alternative_data_daily(_market, market_regime)
│
├─ Phase 2: 銘柄レベルデータ (14銘柄 × 5ツール = 70回)
│   ├─ FinnhubFetcher.fetch_eps_estimates(ticker)
│   ├─ FinnhubFetcher.fetch_recommendation_trends(ticker)
│   ├─ FinnhubFetcher.fetch_price_target(ticker)
│   ├─ FinnhubFetcher.fetch_upgrade_downgrade(ticker)
│   └─ FinnhubFetcher.fetch_company_news(ticker)
│   ※ Rate limit: 55回/分 → 14銘柄×5 = 70回 ≈ 2分で完了
│
├─ Phase 3: インサイダーデータ (SEC は Rate limit なし)
│   └─ EdgarFetcher.fetch_insider_transactions(ticker) × 14銘柄
│
└─ Phase 4: DB保存
    ├─ alternative_data (snapshot) を UPSERT
    └─ alternative_data_daily に INSERT (date = today)
```

### 4.2 スクリプト

```python
# scripts/daily_alt_data.py

def main():
    """日次オルタナティブデータ蓄積"""
    db = QTPDatabase(Path("data/qtp.db"))
    tickers = load_universe("configs/phase5_optimized.yaml")

    # Phase 1: Market-level
    fg = FearGreedFetcher().fetch()
    db.upsert_alternative("_market", "fear_greed", fg)
    db.insert_daily("_market", "fear_greed", date.today(), fg)

    # Phase 2: Finnhub (rate-limited)
    fh = FinnhubFetcher()
    for ticker in tickers:
        for method_name in ["eps_estimates", "recommendation_trends",
                            "price_target", "upgrade_downgrade", "company_news"]:
            data = getattr(fh, f"fetch_{method_name}")(ticker)
            tool = f"finnhub_{method_name}"
            db.upsert_alternative(ticker, tool, data)
            db.insert_daily(ticker, tool, date.today(), data)
            time.sleep(1.1)  # Rate limit safety

    # Phase 3: SEC EDGAR (no rate limit)
    edgar = EdgarFetcher()
    for ticker in tickers:
        data = edgar.fetch_insider_transactions(ticker)
        db.upsert_alternative(ticker, "edgar_insider", data)
        db.insert_daily(ticker, "edgar_insider", date.today(), data)
```

### 4.3 Makefile ターゲット

```makefile
alt-data:  ## Fetch daily alternative data (Finnhub + EDGAR + F&G)
	$(PYTHON) scripts/daily_alt_data.py

alt-data-backfill:  ## Backfill alternative data history (EDGAR insider only)
	$(PYTHON) scripts/backfill_alt_data.py --source edgar --years 3
```

---

## 5. Tier5 特徴量の再設計

### 5.1 新しい特徴量マッピング

| 特徴量名 | ソース | 計算方法 | 時系列化 |
|---------|--------|---------|---------|
| `insider_net_buy_90d` | EDGAR Form 4 | 90日間の (買い件数 - 売り件数) | ✅ 日毎に rolling |
| `insider_buy_value_ratio` | EDGAR Form 4 | 買い金額 / (買い+売り金額) | ✅ 日毎に rolling |
| `analyst_consensus_score` | Finnhub reco trends | strongBuy×2 + buy×1 - sell×1 - strongSell×2 | ✅ 日次蓄積 |
| `analyst_consensus_change` | Finnhub reco trends | 今週スコア - 先週スコア | ✅ 差分 |
| `target_price_upside` | Finnhub price target | (targetMean - currentPrice) / currentPrice | ✅ 日次蓄積 |
| `target_price_revision` | Finnhub price target | 今週targetMean - 先週targetMean | ✅ 差分 |
| `eps_estimate_revision` | Finnhub EPS estimates | 今週EPS予想 - 先週EPS予想 | ✅ 差分 |
| `news_volume_7d` | Finnhub news | 7日間の記事数 | ✅ rolling |
| `news_sentiment_7d` | Finnhub news | 7日間の sentiment 平均 (headline NLP) | ✅ rolling |
| `fear_greed_score` | CNN F&G | 0-100 スコア | ✅ 日次 |
| `fear_greed_change_7d` | CNN F&G | 今日 - 7日前 | ✅ 差分 |
| `upgrade_momentum_30d` | Finnhub upgrade/downgrade | 30日間の (格上げ - 格下げ) 件数 | ✅ rolling |

### 5.2 ポイントインタイム保証

```python
def insider_net_buy_90d(df: pl.DataFrame) -> pl.Series:
    """各日付で「その日時点の過去90日インサイダー買い/売り」を計算"""
    dates = df["date"].to_list()
    ticker = _infer_ticker(df)
    transactions = edgar_fetcher.fetch_insider_history(ticker, years=3)

    result = []
    for d in dates:
        # d 以前の90日間のトランザクションだけカウント
        window_start = d - timedelta(days=90)
        buys = sum(1 for t in transactions if window_start <= t["date"] <= d and t["type"] == "BUY")
        sells = sum(1 for t in transactions if window_start <= t["date"] <= d and t["type"] == "SELL")
        result.append(float(buys - sells))

    return pl.Series("insider_net_buy_90d", result, dtype=pl.Float64)
```

**重要**: Tier6と同じパターン — 各行の日付に対して「その時点で知り得た情報」のみを使用。

### 5.3 蓄積前 vs 蓄積後

| 状態 | insider (EDGAR) | analyst (Finnhub) | news (Finnhub) | F&G |
|------|----------------|-------------------|----------------|-----|
| **即日** (Day 1) | ✅ 過去3年の取引履歴 | ❌ 今日のスナップショットのみ | ❌ 7日分のみ | ✅ 1年分の履歴 |
| **30日後** | ✅ + 30日の日次蓄積 | ✅ 30日のトレンド計算可能 | ✅ 30日のセンチメント推移 | ✅ |
| **90日後** | ✅ フル機能 | ✅ フル機能 | ✅ フル機能 | ✅ |

---

## 6. 実装フェーズ

### Phase A: 即効性（Day 1 で効果あり）

| タスク | 工数 | 効果 |
|--------|------|------|
| A1. EdgarTools インストール + EdgarFetcher 実装 | 2h | インサイダー全履歴が学習に使える |
| A2. fear-greed インストール + FearGreedFetcher 実装 | 1h | F&G 1年分が学習に使える |
| A3. insider_net_buy_90d 特徴量を Tier5 に追加 | 1h | 即座に学習に寄与 |
| A4. fear_greed_score 特徴量を Tier5 に追加 | 0.5h | 即座に学習に寄与 |
| A5. Tier5 テスト追加 | 1h | リグレッション防止 |

### Phase B: 日次蓄積基盤（30日後に効果）

| タスク | 工数 | 効果 |
|--------|------|------|
| B1. Finnhub SDK インストール + FinnhubFetcher 実装 | 2h | アナリスト/EPS/ニュースの安定取得 |
| B2. daily_alt_data.py スクリプト | 1.5h | 日次蓄積パイプライン |
| B3. Makefile + cron 設定 | 0.5h | 自動実行 |
| B4. 蓄積ベースの Tier5 特徴量 (analyst_consensus_change 等) | 2h | 30日蓄積後に有効化 |

### Phase C: 学習統合（90日後にフル効果）

| タスク | 工数 | 効果 |
|--------|------|------|
| C1. Finnhub ニュースの NLP センチメント分析 | 3h | キーワード → LLM or transformer |
| C2. 既存 MCP スクレイピングツールの段階的廃止 | 1h | Finviz/OpenInsider 依存除去 |
| C3. 精度検証 + 特徴量選択 | 2h | Tier5 の効果を定量評価 |

---

## 7. 依存パッケージ

```toml
# pyproject.toml に追加
[project.optional-dependencies]
altdata = [
    "finnhub-python>=2.4",
    "edgartools>=2.0",
    "fear-greed>=0.1",
]
```

### 環境変数

```bash
# .env
FINNHUB_API_KEY=your_free_api_key_here  # https://finnhub.io/register
# EdgarTools: APIキー不要
# fear-greed: APIキー不要
```

---

## 8. リスクと対策

| リスク | 対策 |
|--------|------|
| Finnhub 無料枠の Rate limit (60/分) | RateLimiter クラスで制御。14銘柄×5ツール=70回/実行 ≈ 2分 |
| EdgarTools が日本株に対応しない | 日本株は既存の dexter-jp MCP を継続使用 |
| SEC EDGAR のダウンタイム | 24h キャッシュ + graceful fallback (0.0) |
| Finnhub API のスキーマ変更 | レスポンスを dict で保存。パース時に try/except |
| 日次蓄積が途切れた場合 | 欠損日はスキップ。rolling window 計算で自動補間 |

---

## 9. 成功指標

| 指標 | 現在 | 目標 (90日後) |
|------|------|-------------|
| Tier5 ゼロ重要度特徴量 | 11/18 (61%) | 3/12 以下 (25%以下) |
| Tier5 Top20 入り特徴量数 | 0 | 2以上 |
| WF Accuracy (Phase5) | 65.8% | 68%+ |
| alternative_data_daily レコード数 | 20 | 14銘柄 × 7ツール × 90日 = 8,820 |
