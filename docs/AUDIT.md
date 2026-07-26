# Legacy Codebase Audit — `stock_researcher` → `futures-engine` Migration

**Auditor:** Fable (architect/reviewer). **Date:** 2026-07-25.
**Legacy source:** `C:/Users/tomam/stock_researcher_legacy/stock_researcher-claude-stock-prediction-trading-alerts-w4elnp/`
**Verdict up front:** the legacy system is a daily **stock alerting tool**, not a trading system. It has no execution layer, no cost model, no futures support, and its validation, while honestly intentioned, leaks. Roughly **10% of it carries forward** (the indicator library, the offline-test discipline, and three good *concepts*: provider chains, prediction journaling, held-out hypothesis testing). Everything else is rebuilt in the new repo. This matches the directive: migrate from it, don't patch it.

---

## 1. Inventory

~2,600 LOC Python across 16 modules + 12 test files. Python is untyped-ish (some annotations, no mypy), sklearn-based, config via YAML deep-merge.

| Module | What it does | Fate |
|---|---|---|
| `providers.py` | yfinance / Stooq / AlphaVantage / Tiingo / Finnhub fetchers | **Replace** (concept of pluggable providers survives as a typed interface) |
| `data.py` | Provider chain + file cache (CSV/JSON), stale-cache fallback | **Replace** (no point-in-time correctness; see §2.2) |
| `indicators.py` | SMA/EMA/RSI/MACD/Bollinger/ATR/OBV/stochastic/vol — clean vectorized pandas | **Salvage** — port into the feature library, extend for intraday |
| `features.py` | 15 engineered features + fixed-horizon forward-return target | **Partial** — the explicit `FEATURE_COLUMNS` + single-builder pattern is good; features re-derived for futures; labeling replaced |
| `model.py` | sklearn `HistGradientBoostingRegressor` + expanding-window walk-forward | **Replace** — GBT family choice survives (→ LightGBM); validation superseded |
| `patterns.py` | Rule-based candlestick/chart pattern detectors | **Retire** (equity chart-pattern alerting; out of scope) |
| `signals.py` | Weighted composite of 6 components → BUY/SELL/HOLD | **Retire** (alerting composite, not a strategy) |
| `sentiment.py` | Lexicon news-headline scoring | **Retire** (no meaningful equivalent for MES/MNQ) |
| `claude_analyst.py`, `llm.py` | LLM "head analyst" dossier + verdict (weight 0.35 — the dominant signal) | **Retire from all validation/alpha paths** (see §2.9) |
| `discovery.py`, `hypotheses.py`, `playbook.py` | Pattern Lab: LLM proposes threshold rules; app backtests on a 70/30 holdout | **Retire** (overfitting machine despite honest holdout; see §2.6) |
| `journal.py` | Prediction journal; scores calls after horizon passes | **Replace** — concept becomes the experiment/trial logger feeding the DSR trial counter |
| `alerts.py` | Console/email/webhook + change-dedupe | **Retire** (webhook notification may return inside live monitoring, later) |
| `cli.py` | scan/predict/backtest/discover/screen/evaluate/playbook | **Replace** |
| `config.py` | YAML + deep-merged defaults | **Replace** with typed, validated (pydantic) config |
| `tests/` | Fully offline pytest suite on synthetic OHLCV | **Retain the discipline** — offline synthetic-data tests are the standard for the new repo |
| `.github/workflows/` | Scheduled scans; state committed back to repo | **Replace** with CI for tests + look-ahead audit |

---

## 2. Anti-pattern findings (vs. Non-Negotiable Constraints)

### 2.1 yfinance in the research path — Data constraint #1 (violated)
- `requirements.txt:4` — `yfinance>=0.2.40` is a core dependency.
- `providers.py:49-56` — `fetch_prices_yfinance` is the **first** provider in the default chain (`config.py:19`), with `auto_adjust=True`: retroactively adjusted prices, silently different history on every fetch.
- Every command (`scan`, `backtest`, `discover`, `evaluate`) trains/validates on whatever yfinance returns that day.
- **Disposition:** new system gets a Databento/Norgate-first adapter layer; a yfinance dev-fetcher may exist behind the same interface, hard-marked `NOT FOR VALIDATION` and rejected by the research pipeline at runtime.

### 2.2 No point-in-time correctness — Data constraint #4 (violated)
- `data.py:71-99` — cache is a single mutable CSV per ticker, overwritten on refresh; no snapshots, no hashes, no as-of versioning. Yesterday's backtest is unreproducible today.
- `providers.py:41-44` — `_trim` anchors history to `pd.Timestamp.today()`; `journal.py`, `hypotheses.py` stamp `date.today()`. Runs are wall-clock-dependent.
- `data.py:94-98` — silent fallback to a **stale cache** on provider failure: the dataset you validated on is whatever happened to be on disk.
- **Disposition:** immutable snapshot store (parquet + content hash recorded in every run manifest); CI look-ahead shift-audit.

### 2.3 No futures, no continuous contracts — Data constraint #3 (absent)
- Equity tickers and sector ETFs only. Nothing in the codebase knows what a roll, a contract multiplier, or a tick size is.
- **Disposition:** built from scratch — explicit continuous-contract builder (volume/OI or calendar roll; Panama difference vs ratio adjustment; method + roll dates recorded in dataset metadata; loud failure on raw spliced input).

### 2.4 Fixed-horizon labels, iid-treated overlapping samples — Modeling constraints #2/#3 (violated)
- `features.py:72` — the only label is `close.shift(-horizon)/close - 1` (5-day fixed horizon).
- `model.py` — overlapping 5-day-return rows are fed to the GBT as independent samples; no uniqueness weighting, no meta-labeling, no path-dependent barriers.
- **Disposition:** triple-barrier labeling + meta-labeling + uniqueness weights; fixed-horizon kept only as a documented baseline.

### 2.5 Walk-forward with boundary leakage; no purging/embargo/CPCV — Validation constraints (violated)
- `model.py:73-84` — expanding-window folds where training rows run right up to `train_end`. With a 5-day forward-return label, the last ~5 training labels are computed from prices **inside the test window**. This is precisely the leak purging exists to remove.
- No embargo, no CPCV, no multiple-testing accounting. (Plain k-fold is at least absent.)
- **Disposition:** dedicated splitter module (purged k-fold + embargo, walk-forward, CPCV) with label-interval awareness; leak tests in CI.

### 2.6 No overfitting controls; the Pattern Lab is a multiplicity machine — Validation constraints (violated)
- `hypotheses.py:142-155` — `passes_validation`: `n_triggers >= 8`, `lift >= 1.2`, hit-rate ≥ 0.45. No significance test, no correction for the number of hypotheses tried.
- `discovery.py` — an LLM proposes up to 5 rules per ticker per weekly run, forever, each tested on the *same* fixed 30% holdout. The holdout stops being out-of-sample after the first few dozen queries against it. The playbook records rejections (honest!) but nothing computes the implied trial count.
- No DSR, no PBO, no bootstrap CIs anywhere.
- **Disposition:** every configuration evaluated anywhere in the new system writes a trial record; DSR/PBO/bootstrap CIs are mandatory report outputs; red-flag reporter (Sharpe > 3, win-rate > 70% + smooth equity, edge that dies with 1-bar delay or costs).

### 2.7 Zero transaction costs, zero execution realism — Backtesting constraints (violated)
- `hypotheses.backtest` counts hit rates on forward returns. `model.py` reports MAE/hit-rate. Nothing anywhere models commissions, fees, spread, slippage, or fill timing.
- Signals are computed **on the close of the same bar** that would have to be traded (`signals.py:241`, `journal.py:record` uses `last_price` = signal-bar close): an implicit look-ahead fill assumption with no delay option.
- **Disposition:** parameterized cost model (commission/RT, exchange+NFA fees, spread, slippage, 1-bar delay option); all reported results net; mandatory gross-vs-net report; two-stage vectorized → event-driven (Nautilus) stack.

### 2.8 No execution, OMS, risk, or kill switches — Risk & execution constraints (absent)
- The "action" layer is `alerts.py` (email/webhook). There is no order, position, account, or risk concept in the codebase.
- **Disposition:** built from scratch per the target architecture (OMS, broker adapters behind the backtest-execution interface, non-overridable kill switches, reconciliation, monitoring).

### 2.9 LLM-as-alpha in the signal path — Modeling constraints (violated in spirit and letter)
- `config.py:63-70` — Claude's verdict carries the **largest weight (0.35)** in the composite; `signals.py:215` blends it directly into the score.
- An LLM verdict is not point-in-time replayable, not backtestable, not stationary across model versions — it cannot exist in a validated strategy. (The legacy README's own candor supports this.)
- **Disposition:** no LLM output in any research, validation, or execution path. LLM-assisted *research tooling* (e.g. code review, report narration) is fine; LLM-generated *signals* are not.

### 2.10 Engineering-standard violations
- Magic constants in strategy logic: `signals.py:118` (`_squash(x, 0.03)`), `signals.py:122` (edge window 0.45/0.15), pattern strengths throughout `patterns.py`, validation bar in `hypotheses.py:146-149`.
- Reproducibility: wall-clock dependence everywhere (§2.2); no run manifests, no seeds policy (a lone `random_state=42`), no experiment logging.
- Typing/linting: partial annotations, no mypy/ruff, no CI quality gates.
- **Disposition:** new-repo standards — Python 3.11+, mypy --strict on library code, ruff, pydantic-validated config, every run emits a manifest (config hash, data snapshot hash, seed, git SHA, trial IDs).

---

## 3. What genuinely survives

1. **`indicators.py`** — correct, clean, vectorized; Wilder RSI/ATR done properly. Ported as the seed of the feature library.
2. **Offline synthetic-data test suite discipline** — the entire legacy suite runs with no network. This is the testing standard for the new repo.
3. **Concepts:** provider chains (→ typed adapter interface), prediction journal (→ trial/experiment logger), "LLM proposes / code verifies on data it never saw" (→ documented future research tooling only, never in the validation loop), GBT-on-engineered-features (→ LightGBM primary).

## 4. Explicitly out of scope for the new system

News sentiment, analyst consensus, equity screener, chart-pattern alerting, email alerts, the Pattern Lab, and all LLM-in-the-loop signal generation. None of these have a defensible role in a validated MES/MNQ futures system under the research constraints.
