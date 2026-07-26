# Futures Engine Migration & Build Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. **Orchestration protocol for this project:** Fable (architect) assigns tasks, reviews every deliverable against the task's Acceptance Criteria and the Global Constraints, and iterates with written feedback until APPROVED. Opus subagents write all code. Fable writes none — task specs below therefore give exact interfaces and behavioral test specifications instead of literal code; implementers author the code and tests.

**Goal:** Rebuild the legacy yfinance stock-alert codebase into a validated research-and-execution system for systematic trend/momentum trading of CME micro futures (MES/MNQ), targeting prop-firm funded accounts (Topstep/MFFU) via API automation.

**Architecture:** Layered package `futures_engine`: point-in-time data layer (provider adapters → immutable snapshots → continuous contracts) → feature/labeling library (triple-barrier, meta-labeling, regimes) → two-stage backtest stack (vectorized triage → Nautilus Trader event-driven validation, both net-of-costs) → validation statistics (purged CV/CPCV, DSR/PBO, red flags) → prop-survival Monte Carlo → execution layer (OMS, broker adapters, non-overridable kill switches) → single per-run research report with go/no-go verdict. Every evaluated configuration is logged to a trial store that feeds the DSR trial counter.

**Tech Stack:** Python 3.11+, pandas/numpy, LightGBM (+ scikit-learn regularized linear baselines), hmmlearn + ruptures (regimes), vectorbt OSS (or thin internal vectorized harness — see T5), Nautilus Trader (event-driven), databento + norgatedata clients, pydantic v2 + YAML config, pyarrow/parquet storage, pytest (+ hypothesis for property tests), ruff + mypy, Tradovate REST/WS (live, later ProjectX/Rithmic).

## Global Constraints (enforced in every Fable review)

Copied from the research constraints; violations are automatic review rejections.

- **G1** No yfinance in any research/backtest/validation path. Dev fetcher only, behind the provider interface, `validation_grade=False`, marked `NOT FOR VALIDATION`; pipeline refuses such data at runtime.
- **G2** All vendor access behind the provider interface; no vendor calls in strategy/feature/backtest code. Config-driven provider selection.
- **G3** Continuous contracts built explicitly: roll rule ∈ {volume, open_interest, calendar}, adjustment ∈ {panama_diff, ratio, none}; method + roll dates recorded in `DatasetMeta`; backtests fail loudly on futures data lacking `ContinuousMeta`.
- **G4** Point-in-time everywhere: immutable snapshots with content hashes; no feature uses information unavailable at decision time; automated shift-test look-ahead audit in CI.
- **G5** Models: LightGBM/XGBoost + regularized linear baselines only. No deep nets without a gated OOS comparison beating the boosted baseline. No RL for alpha. No LLM output in any signal/validation/execution path.
- **G6** Labels: triple-barrier + meta-labeling primary; fixed-horizon only as documented baseline. Uniqueness sample weights. Regime module available to gate strategies.
- **G7** Strategy family: trend/momentum. HF mean reversion out of scope.
- **G8** Every reported result net of costs (commissions ~$1.02–$1.04/RT micros parameterized per instrument, exchange+NFA fees, spread, slippage model, optional 1-bar delay). Gross-vs-net always reported.
- **G9** CV: purged k-fold with embargo, walk-forward, CPCV. Plain k-fold forbidden. Splitters are label-interval-aware.
- **G10** Mandatory outputs: DSR with honest trial count from the trial store, PBO, bootstrap/MC Sharpe CIs, red-flag warnings (Sharpe > 3; win rate > 70% with smooth equity; edge vanishing under 1-bar delay or realistic costs).
- **G11** Prop survival gate: Monte Carlo vs configurable prop rules (trailing DD intraday-unrealized and EOD variants, daily loss limit, consistency rule); >90% survival required for any live recommendation.
- **G12** Sizing: volatility-targeted, capped ≤ quarter-Kelly, always dominated by the survival constraint.
- **G13** Kill switches enforced in the execution layer before orders leave the system; impossible for a strategy to override (strategies never hold a broker client reference). `isAutomated: true` on all automated orders (CME Rule 575).
- **G14** Backtest–live parity: live broker adapters implement the same `ExecutionClient` interface as backtest execution.
- **G15** Engineering: Python 3.11+, typed (mypy clean on library code), ruff clean, pydantic-validated config (no magic constants in strategy code), reproducible runs (seed policy, data snapshot hash, config hash, git SHA in a `RunManifest`), all tests offline (recorded fixtures/synthetic data — no network in tests).
- **G16** Highest-stakes components get the hardest tests: cost model, CV splitters, survival simulator, kill switches.

## Shared Contracts (authoritative names/signatures — all tasks code against these)

Types below are the cross-task API. Implementers may add internals freely but must not rename or reshape these without Fable approval.

```text
futures_engine/core/types.py
  InstrumentSpec(symbol_root: str, exchange: str, tick_size: float, tick_value: float,
                 multiplier: float, currency: str)                        # MES: 0.25/$1.25/$5; MNQ: 0.25/$0.50/$2
  BarInterval = Literal["1m", "5m", "15m", "1h", "1d"]
  ContinuousMeta(roll_rule: Literal["volume","open_interest","calendar"],
                 adjustment: Literal["panama_diff","ratio","none"],
                 roll_dates: list[date], underlying_contracts: list[str])
  DatasetMeta(symbol_root: str, source: str, interval: BarInterval, start: datetime, end: datetime,
              continuous: ContinuousMeta | None, snapshot_hash: str, as_of: datetime,
              validation_grade: bool)
  Bars = pd.DataFrame  # UTC DatetimeIndex; columns open, high, low, close, volume (float64)

futures_engine/core/manifest.py
  RunManifest(run_id: str, created_at: datetime, git_sha: str, config_hash: str,
              data_snapshot_hashes: list[str], seed: int, trial_ids: list[str])

futures_engine/data/provider.py
  class MarketDataProvider(Protocol):
      name: str
      validation_grade: bool
      def fetch_bars(contract: str, start: datetime, end: datetime, interval: BarInterval) -> Bars
      def list_contracts(symbol_root: str, start: date, end: date) -> list[ContractInfo]
  ContractInfo(symbol: str, expiry: date, first_trade: date | None)

futures_engine/data/store.py
  class SnapshotStore:
      def save(bars: Bars, meta: DatasetMeta) -> str            # returns snapshot_hash; immutable
      def load(snapshot_hash: str) -> tuple[Bars, DatasetMeta]
  def require_validation_grade(meta: DatasetMeta) -> None       # raises DataIntegrityError on dev-grade
                                                                # or futures data with continuous=None

futures_engine/data/continuous.py
  def build_continuous(per_contract: dict[str, Bars], specs: list[ContractInfo],
                       roll_rule, adjustment) -> tuple[Bars, ContinuousMeta]

futures_engine/costs/model.py
  CostConfig(commission_per_side_usd: float, exchange_fee_per_side_usd: float,
             nfa_fee_per_side_usd: float, spread_ticks: float,
             slippage: Literal["fixed_ticks","vol_scaled"], slippage_ticks: float,
             delay_bars: int)                                   # delay_bars in {0, 1}
  TradeLog = pd.DataFrame  # entry_ts, exit_ts, side, qty, entry_px, exit_px, gross_pnl_usd
  def apply_costs(trades: TradeLog, spec: InstrumentSpec, cfg: CostConfig) -> TradeLog
      # adds: commission_usd, fees_usd, spread_cost_usd, slippage_usd, net_pnl_usd

futures_engine/prop/rules.py
  PropRuleSet(name: str, start_balance: float, trailing_dd: float,
              trailing_mode: Literal["intraday_unrealized","eod"],
              trailing_freezes_at_start_balance: bool, daily_loss_limit: float | None,
              consistency_max_day_pct: float | None, profit_target: float,
              min_trading_days: int)
  presets(): topstep_50k, mffu_50k, apex_50k (values in YAML, not code)
  def simulate_account(day_pnl_paths | trade_seq, rules: PropRuleSet) -> AccountOutcome
  AccountOutcome(survived: bool, hit_target: bool, bust_reason: str | None, days_elapsed: int)

futures_engine/prop/survival.py
  def monte_carlo_survival(trades: TradeLog, rules: PropRuleSet, contracts: int,
                           n_paths: int, horizon_days: int, block_size: int, seed: int) -> SurvivalReport
  SurvivalReport(p_survival: float, p_target_before_bust: float, median_days_to_target: float | None,
                 bust_reasons: dict[str, float], ci_90: tuple[float, float])

futures_engine/validation/splitters.py
  LabelIntervals = pd.Series  # index=t0 (event start), values=t1 (label end)
  class PurgedKFold(n_splits: int, embargo_frac: float): split(X, intervals) -> Iterator[(train_idx, test_idx)]
  class CombinatorialPurgedCV(n_groups: int, n_test_groups: int, embargo_frac: float)
  class WalkForward(n_folds: int, min_train: int, purge: bool = True)

futures_engine/validation/stats.py
  def deflated_sharpe(observed_sr: float, n_trials: int, n_obs: int, skew: float, kurt: float) -> float
  def pbo(perf_matrix: np.ndarray, n_partitions: int) -> float     # rows=configs, cols=time slices
  def bootstrap_sharpe_ci(returns: pd.Series, n_boot: int, block_size: int, seed: int,
                          conf: float) -> tuple[float, float]
  def red_flags(result: BacktestResult, delayed: BacktestResult | None,
                gross: BacktestResult | None) -> list[RedFlag]
  RedFlag(code: str, message: str, severity: Literal["warn","fail"])

futures_engine/trials/logger.py
  TrialRecord(trial_id: str, run_id: str, ts: datetime, strategy_family: str, config_hash: str,
              params: dict, data_snapshot_hashes: list[str], cv_scheme: str,
              metrics: dict[str, float], seed: int, git_sha: str)
  class TrialLogger:  # append-only SQLite
      def log(record: TrialRecord) -> None
      def count(strategy_family: str | None = None) -> int        # -> DSR n_trials
      def all(strategy_family: str | None = None) -> list[TrialRecord]

futures_engine/labels/triple_barrier.py
  def triple_barrier(bars: Bars, events: pd.DatetimeIndex, pt_mult: float, sl_mult: float,
                     max_holding_bars: int, vol: pd.Series) -> Labels
  Labels = pd.DataFrame  # index=t0; cols: t1, label {-1,0,1}, ret, touch {"pt","sl","time"}
  def meta_labels(primary_side: pd.Series, labels: Labels) -> pd.Series  # binary
  def uniqueness_weights(labels: Labels, bar_index: pd.DatetimeIndex) -> pd.Series

futures_engine/regime/detector.py
  class RegimeDetector(Protocol): fit(bars) -> Self; regimes(bars) -> pd.Series[int];
                                  proba(bars) -> pd.DataFrame
  HMMRegimeDetector(n_states, seed); ChangePointRegimeDetector(model, penalty)

futures_engine/research/harness.py
  class VectorSignal(Protocol):
      family: str
      def generate(bars: Bars, params: dict) -> pd.Series      # float target position in [-1, 1]
  def sweep(signal: VectorSignal, grid: dict[str, list], snapshot_hashes: list[str],
            cost_cfg: CostConfig, splitter, logger: TrialLogger, seed: int) -> SweepResult
  SweepResult(table: pd.DataFrame, best: dict, n_trials_logged: int)

futures_engine/research/meta_model.py
  MetaModelPipeline(primary: VectorSignal, model: Literal["lightgbm","logistic_l2"],
                    params: dict, seed: int)
      def fit(bars: Bars, labels: Labels, weights: pd.Series, splitter) -> FitResult
      def predict(bars: Bars) -> pd.Series          # p(trade) in [0,1], used for gate + sizing input
  FitResult(oos_metrics: dict, per_fold: pd.DataFrame, model_artifact: Path)

futures_engine/sizing/position.py
  SizingConfig(target_daily_vol_usd: float, kelly_fraction_cap: float,   # cap ≤ 0.25 enforced
               max_contracts: int)
  def position_size(vol_estimate: float, edge: EdgeStats, spec: InstrumentSpec,
                    cfg: SizingConfig, survival_max_contracts: int) -> int
      # min(vol-target size, fractional-Kelly size, survival_max_contracts, max_contracts)
  EdgeStats(p_win: float, avg_win: float, avg_loss: float)

futures_engine/backtest/engine.py
  class BacktestRunner:  # Nautilus wrapper
      def run(snapshot_hash: str, strategy_cfg: StrategyConfig, cost_cfg: CostConfig,
              spec: InstrumentSpec) -> BacktestResult
  BacktestResult(trades: TradeLog, equity: pd.Series, returns: pd.Series,
                 fills: pd.DataFrame, metrics: dict[str, float], manifest: RunManifest)

futures_engine/execution/client.py
  Order(client_order_id: str, instrument: str, side: Literal["buy","sell"], qty: int,
        type: Literal["market","limit","stop"], limit_px: float | None, stop_px: float | None,
        is_automated: bool = True)                                # CME Rule 575 — always True here
  class ExecutionClient(Protocol):   # implemented by backtest sim AND live adapters
      def submit(order: Order) -> OrderAck
      def cancel(client_order_id: str) -> None
      def positions() -> list[Position];  def account() -> AccountState
      def on_disconnect(cb) / on_data_stale(cb)

futures_engine/execution/risk.py
  class RiskManager:      # owns the ExecutionClient; strategies submit through OMS -> RiskManager only
      def approve(order: Order, state: AccountState) -> Approval  # Approval(ok, reason)
      kill switches (each independently testable, config-driven):
        daily_loss_limit(buffer_usd), trailing_dd_guard(margin_usd), stale_data_halt(max_age_s),
        flatten_on_disconnect(), max_order_rate(n_per_minute)

futures_engine/report/builder.py
  def build_report(run_id: str, logger: TrialLogger, survival: SurvivalReport,
                   result: BacktestResult, gates: GateConfig) -> Path   # writes Markdown+HTML artifact
  GateConfig(min_dsr_p: float, max_pbo: float, min_survival: float)     # defaults 0.95 / 0.5 / 0.90
  Verdict: GO only if all gates pass and no severity="fail" red flags
```

## Architect decisions already made (do not re-litigate in tasks)

1. **Legacy custom backtester is replaced, not refactored.** `hypotheses.py` is a hit-rate counter with no notion of position, cost, or path; parity-refactoring it would cost more than adopting Nautilus. (Constraint "Fable decides": decided.)
2. **Nautilus Trader** is the event-driven engine (Windows wheels exist for Py 3.11+). If the T6 implementer hits a hard blocker, escalate to Fable — do not silently substitute.
3. **vectorbt OSS preferred for T5**; if licensing/API friction proves material, implement a thin internal vectorized harness behind the same `sweep()` interface and document the decision in the self-report. Either way the `sweep()` contract holds.
4. **Package name `futures_engine`, repo `futures-engine`.** Legacy code stays outside the repo at `C:/Users/tomam/stock_researcher_legacy/` (read-only reference).
5. **Salvage list** (port, don't import): `indicators.py` (into T4's feature lib), offline-synthetic-test discipline (all tasks), explicit-feature-registry pattern (T4).
6. **Databento is the primary historical source** (GLBX.MDP3, MES/MNQ, 1-min + daily), **Norgate** for survivorship-free EOD cross-checks. All adapter tests run offline against recorded fixtures; live API calls never occur in CI.

---

## Task Graph & Waves

```
Wave 0:  T0 scaffold ──────────────────────────────────────────────┐
Wave 1:  T1 data layer      T2 costs+prop rules      T3 validation │  (parallel; only T0 required)
Wave 2:  T4 features/labels/regime (needs T1)                      │
         T5 vectorized research harness (needs T1,T2,T3; meta-model half needs T4) │
Wave 3:  T6 event-driven engine (needs T1,T2,T5)                   │
         T7 survival simulator (needs T2; consumes T6 TradeLogs)   │
Wave 4:  T8 execution & risk layer (needs T6 interfaces)           │
         T9 reporting + end-to-end + README (needs all)            ┘
```

Serialization: T1 before T4/T5/T6 (data schemas), T2 before T5/T7 (costs/rules), T6 before T8 (shared `ExecutionClient`). T2 and T3 are pure-logic tasks that parallelize perfectly with T1.

---

### Task T0: Repo scaffold, typed config, trial logger, CI

**Files:** Create `pyproject.toml`, `futures_engine/{__init__,core/types,core/manifest,core/config}.py`, `futures_engine/trials/logger.py`, `configs/{instruments.yaml,costs.yaml,prop_rules.yaml,example_strategy.yaml}`, `tests/test_config.py`, `tests/test_trial_logger.py`, `.github/workflows/ci.yml`, `.gitignore`, `ruff.toml`/mypy config.

**Interfaces:** Produces `InstrumentSpec`, `RunManifest`, `TrialLogger`, pydantic `Settings` loader (YAML → validated models; unknown keys rejected). Consumes nothing.

**Acceptance criteria:**
- [ ] `pip install -e .[dev]` works on Windows, Python 3.11+; ruff + mypy clean; pytest green offline.
- [ ] Config: loading `configs/instruments.yaml` yields `InstrumentSpec` for MES/MNQ with correct tick economics; invalid/unknown keys raise validation errors (tested).
- [ ] TrialLogger: append-only (no update/delete API); `count()` correct across process restarts; concurrent writes safe (tested with threads); every record requires `data_snapshot_hashes`, `config_hash`, `seed`, `git_sha` — missing fields rejected (tested).
- [ ] `RunManifest` round-trips to JSON; manifest creation pulls git SHA from the repo.
- [ ] CI workflow runs ruff, mypy, pytest; includes a placeholder job `lookahead-audit` that T1 will fill.
- [ ] Self-report lists assumptions + limitations.

### Task T1: Point-in-time data layer

**Files:** Create `futures_engine/data/{provider,store,continuous,audit}.py`, `futures_engine/data/adapters/{databento_adapter,norgate_adapter,yfinance_dev}.py`, `tests/data/{test_store,test_continuous,test_providers,test_lookahead_audit}.py`, `tests/fixtures/` (recorded/synthetic per-contract bar fixtures covering ≥3 roll events).

**Interfaces:** Consumes T0 types/config. Produces `MarketDataProvider`, `SnapshotStore`, `build_continuous`, `require_validation_grade`, `DataIntegrityError`.

**Acceptance criteria:**
- [ ] Adapters implement the protocol; **no vendor import outside `adapters/`** (enforced by a test that greps the package). Databento/Norgate adapters run against recorded fixtures offline; network code isolated and mockable.
- [ ] yfinance dev fetcher: `validation_grade=False`, module docstring `NOT FOR VALIDATION`; `require_validation_grade` raises on it (tested), and raises on futures bars whose meta lacks `ContinuousMeta` (G3, tested).
- [ ] SnapshotStore: parquet + sidecar meta; hash = content hash (same data → same hash, any mutation → different hash, tested); `load` returns bit-identical frames; snapshots are write-once (attempted overwrite raises).
- [ ] Continuous builder: volume, open-interest, and calendar roll rules; panama_diff, ratio, none adjustments; roll dates + underlying contracts recorded; **property test**: panama-adjusted series has zero price jump at rolls while raw splice shows the gap; ratio adjustment preserves returns across rolls.
- [ ] Look-ahead audit (`audit.py`): shift test — recompute any registered feature/dataset with input history truncated at t and assert values at t are unchanged; wired into CI job from T0. Deliberately leaky sample feature is caught (tested).
- [ ] All timestamps UTC; DST/holiday roll edge covered by at least one fixture.

### Task T2: Cost model + prop rule engine

**Files:** Create `futures_engine/costs/model.py`, `futures_engine/prop/rules.py`, `configs/costs.yaml` (MES/MNQ commissions ~$1.02–$1.04/RT parameterized, exchange+NFA fees), `configs/prop_rules.yaml` (Topstep/MFFU/Apex presets), `tests/{test_costs,test_prop_rules}.py`.

**Interfaces:** Consumes T0 `InstrumentSpec`, config. Produces `CostConfig`, `apply_costs`, `TradeLog` schema, `PropRuleSet`, `simulate_account`, `AccountOutcome`.

**Acceptance criteria (G16 — test hardest):**
- [ ] `apply_costs` per-trade decomposition: commission, fees, spread (ticks→USD via tick_value), slippage (fixed and vol-scaled), each hand-verified in tests against worked examples for MES and MNQ; `net = gross − Σcosts` exactly.
- [ ] `delay_bars=1` shifts execution price to next bar's open (consumed by T5/T6; the semantic is defined and tested here on a reference TradeLog).
- [ ] Rule engine: trailing DD in both `intraday_unrealized` and `eod` modes with tests distinguishing them on a crafted path that survives EOD-trailing but busts intraday-trailing; trailing freeze at start balance when configured; daily loss limit breach ends the day flat; consistency rule (max single-day % of total profit) evaluated at payout; boundary cases (exact-touch of limits) covered by explicit tests.
- [ ] Presets load from YAML only — zero rule numbers in code (G15). Preset values carry source comments/dates in YAML.
- [ ] Property test: for any pnl path, busting is monotone in trailing_dd (tighter DD never survives a path a looser DD busted on... i.e. if account busts at DD=X it must also bust at DD<X).

### Task T3: Validation & statistics module

**Files:** Create `futures_engine/validation/{splitters,stats}.py`, `tests/validation/{test_splitters,test_dsr_pbo,test_bootstrap,test_red_flags}.py`.

**Interfaces:** Consumes T0 (TrialLogger for n_trials). Produces `PurgedKFold`, `CombinatorialPurgedCV`, `WalkForward`, `deflated_sharpe`, `pbo`, `bootstrap_sharpe_ci`, `red_flags`.

**Acceptance criteria (G16):**
- [ ] PurgedKFold: with overlapping label intervals, **no training sample's interval overlaps any test interval** (exhaustively asserted in tests); embargo removes the trailing fraction after each test block; a plain-k-fold-equivalent configuration is impossible to construct (purging always on).
- [ ] CPCV: all C(n_groups, n_test) combinations produced; per López de Prado — number of paths and group assignments verified for a small case by hand in tests.
- [ ] `deflated_sharpe`: reproduces a published worked example within tolerance; monotone decreasing in n_trials (property test).
- [ ] `pbo`: symmetric-noise sanity check → PBO ≈ 0.5 on pure noise (statistical tolerance, seeded); ≈ 0 for a genuinely dominant config.
- [ ] `bootstrap_sharpe_ci`: stationary block bootstrap; seeded/reproducible; coverage sanity test on synthetic iid returns.
- [ ] `red_flags`: fires on Sharpe > 3; win rate > 70% with smooth equity (definition: max drawdown < k·vol, k in config); edge that vanishes with delay/costs (compares provided delayed/gross variants); each flag has its own test.
- [ ] Leakage regression test: walk-forward with `purge=True` shows measurably lower (or equal) skill than unpurged on an overlapping-label synthetic dataset engineered to leak — this test documents the legacy bug (AUDIT §2.5) and prevents its return.

### Task T4: Feature, labeling & regime library

**Files:** Create `futures_engine/features/{indicators,builder,fracdiff}.py`, `futures_engine/labels/triple_barrier.py`, `futures_engine/regime/detector.py`, `tests/features/…`, `tests/labels/…`, `tests/regime/…`.

**Interfaces:** Consumes T1 (`Bars`, snapshot loading), T0 config. Produces `triple_barrier`, `meta_labels`, `uniqueness_weights`, `frac_diff`, `RegimeDetector` impls, feature registry `build_features(bars, spec, config) -> pd.DataFrame` with explicit registered feature list (legacy `FEATURE_COLUMNS` pattern).

**Acceptance criteria:**
- [ ] Indicators ported from legacy `indicators.py` with tests reproducing legacy values on shared fixtures (guards the port); intraday-aware (no hardcoded 252-day annualization where interval ≠ 1d — annualization factor derives from interval).
- [ ] Every registered feature passes the T1 shift-audit (registered into the CI audit); a feature using `.shift(-1)` is rejected by the audit (tested).
- [ ] Triple-barrier: vertical/PT/SL touch logic verified on hand-constructed paths (all six orderings of touches); vol-scaled barriers use trailing vol estimate only; `touch` column records which barrier hit.
- [ ] `uniqueness_weights`: overlapping labels → weights < 1, disjoint labels → weights = 1 (tested); weights integrate with LightGBM `sample_weight`.
- [ ] `frac_diff`: d=0 identity, d=1 ≈ first difference (tested); memory-vs-stationarity trade-off demonstrated (ADF on fixture).
- [ ] Regime detectors: deterministic under fixed seed; HMM and change-point produce regime series aligned to bar index with **no look-ahead** (fit on past, label at t uses data ≤ t — walk-forward refit or filtered/online inference; asserted by shift test).
- [ ] Fixed-horizon labeler included and explicitly documented as baseline-only (G6).

### Task T5: Vectorized research harness

**Files:** Create `futures_engine/research/{harness,meta_model,strategies/momentum}.py`, `tests/research/…`.

**Interfaces:** Consumes T1 snapshots, T2 `CostConfig`/`apply_costs`, T3 splitters, T4 labels/weights, T0 `TrialLogger`. Produces `VectorSignal`, `sweep`, `SweepResult`, `MetaModelPipeline`, and ≥2 reference trend/momentum signal families (e.g. donchian breakout, MA-cross with vol filter) as the triage battery.

**Acceptance criteria:**
- [ ] `sweep` refuses non-validation-grade snapshots (calls `require_validation_grade`, tested with the yfinance dev fetcher's output).
- [ ] **Every parameter combination evaluated logs exactly one TrialRecord** — test asserts `n_trials_logged == |grid|` and that aborted/errored combos still log (with error metric) — the DSR counter must be honest (G10).
- [ ] Results net of costs by construction; gross retained alongside (columns `sharpe_gross`, `sharpe_net`); 1-bar delay variant computed for every combo (feeds red-flag check).
- [ ] Vectorized PnL cross-checked against a hand-computed toy case (3 trades on synthetic bars, exact USD equality including costs).
- [ ] Runtime: full grid of ≥500 combos on 5 years of 1-min MES fixture data completes in reasonable time (document measured runtime; vectorization required — no per-row Python loops in the hot path; Numba only with profiling evidence, G15).
- [ ] Strategy params come from YAML config; no magic constants in signal code (G15).
- [ ] `MetaModelPipeline` (G5/G6): LightGBM primary + L2-logistic baseline trained on triple-barrier meta-labels with T4 `uniqueness_weights` as `sample_weight`, evaluated **only** through T3 splitters (constructor takes a splitter; plain k-fold impossible); both models' OOS metrics reported side-by-side; deterministic under fixed seed; every fit logs a TrialRecord.
- [ ] Wave-2 note: T5 depends on T4 for labels/weights — the harness/sweep half may start in parallel with T4, but meta-model work merges only after T4 is APPROVED.

### Task T6: Event-driven validation engine (Nautilus)

**Files:** Create `futures_engine/backtest/{engine,strategy_adapter,parity}.py`, `tests/backtest/…`.

**Interfaces:** Consumes T1 snapshots (+meta), T2 costs, T5 signals/models. Produces `BacktestRunner`, `BacktestResult`, the Nautilus `StrategyAdapter`, and the backtest-side `ExecutionClient` implementation (shared contract with T8).

**Acceptance criteria:**
- [ ] Runner ingests only validation-grade snapshots with `ContinuousMeta` (loud failure otherwise, G3 — tested).
- [ ] Fill model: costs/fees via T2 config (never re-implemented inline); spread + slippage applied at fill; `delay_bars=1` option produces strictly different (and on trending fixtures worse-or-equal) results than delay 0 (tested).
- [ ] **Parity test:** reference momentum strategy run through both T5 vectorized and T6 event-driven paths on the same snapshot: trade count within ±1 per 100 trades, net PnL within a stated tolerance; deviations beyond tolerance fail CI. Documented sources of residual difference.
- [ ] `BacktestResult.manifest` fully populated (snapshot hashes, config hash, seed, git SHA); rerun with same manifest inputs is bit-reproducible (metrics equal, tested).
- [ ] Every `BacktestRunner.run` logs a TrialRecord.

### Task T7: Prop-account survival simulator

**Files:** Create `futures_engine/prop/survival.py`, `futures_engine/sizing/position.py`, `tests/prop/test_survival.py`, `tests/sizing/test_position.py`.

**Interfaces:** Consumes T2 (`PropRuleSet`, `simulate_account`), TradeLogs from T6 (or any conforming TradeLog). Produces `monte_carlo_survival`, `SurvivalReport`, `position_size`, `SizingConfig` (consumed by T8 RiskManager and T9 report).

**Acceptance criteria (G16):**
- [ ] Trade-level **block bootstrap** (preserves autocorrelation; block size configurable) over the strategy's TradeLog; intraday mark-to-market path approximated from trade MAE/MFE when available, documented when not.
- [ ] Seeded and reproducible; n_paths configurable; 90% CI on p_survival reported (G11).
- [ ] Validation against analytically known cases: (a) all-winning trades → p_survival = 1; (b) deterministic loss sequence that busts → 0; (c) simple random walk vs closed-form gambler's-ruin approximation within tolerance.
- [ ] Sizing interaction: exposes `contracts` sweep so the report can show survival vs size; test that p_survival is monotone non-increasing in contract count for fixed rules.
- [ ] Gate helper: `passes_gate(report, min_survival=0.90)` used by T9 verdict (G11).
- [ ] Sizing (G12): `position_size` returns the **minimum** of vol-target size, fractional-Kelly size (cap ≤ 0.25 validated at config load — larger cap rejected), `survival_max_contracts` (largest size whose SurvivalReport passes the gate), and `max_contracts`; each leg unit-tested, including that the survival constraint dominates Kelly when they disagree.

### Task T8: Execution & risk layer

**Files:** Create `futures_engine/execution/{client,oms,risk,reconcile,monitor}.py`, `futures_engine/execution/adapters/{tradovate,projectx_stub,rithmic_stub}.py`, `configs/live.yaml` (incl. shutdown criteria), `tests/execution/…`.

**Interfaces:** Consumes shared `ExecutionClient` contract (T6), T2 rules, T7 `position_size`/`SizingConfig`, T0 config. Produces the live **market-data handler** (Tradovate WS feed → normalized ticks/bars driving strategy + staleness clock), OMS, `RiskManager` (which also enforces that order qty ≤ current `position_size` output), Tradovate adapter (paper/demo endpoints), reconciler, monitoring hooks (rolling live-vs-backtest slippage, fill quality, rolling Sharpe, drift stats). Live event flow: market data handler → strategy engine → OMS → RiskManager → ExecutionClient, with state persistence at each hop.

**Acceptance criteria (G13/G14/G16 — test hardest):**
- [ ] Architecture enforces non-overridability **by construction**: strategies receive an OMS handle only; the `ExecutionClient` is owned by the OMS; every order passes `RiskManager.approve` before submission — test proves a strategy cannot reach the client (no public accessor; attempted bypass in test fails at type/runtime level).
- [ ] Kill switches each unit-tested: daily loss limit with buffer (order rejected + flatten triggered at breach), trailing-DD guard margin, stale-data halt (no fresh tick within max_age → halt + flatten), flatten-on-disconnect (simulated WS drop → flatten orders queued for reconnect + alarm), max order rate (burst rejected). Switch state transitions logged.
- [ ] Idempotency: duplicate `client_order_id` submits are no-ops; crash-restart replays outbox without double-sending (state persistence tested with a kill-restart simulation).
- [ ] Reconciliation: on reconnect, broker-state fixture differing from local state (extra fill, missed fill, position mismatch) is detected and corrected; each mismatch class tested.
- [ ] Every outbound order has `is_automated=True` (CME 575) — asserted in adapter serialization tests.
- [ ] Tradovate adapter tested against recorded HTTP/WS fixtures only (no network in CI); ProjectX/Rithmic are typed stubs raising `NotImplementedError` with the interface in place.
- [ ] Shutdown criteria (max live-vs-backtest slippage divergence, rolling-Sharpe floor, drift thresholds) read from `configs/live.yaml`, not code (constraint: shutdown criteria in config).

### Task T9: Reporting, end-to-end pipeline, README

**Files:** Create `futures_engine/report/builder.py`, `futures_engine/pipeline/run.py` (single-command research run), `examples/mes_momentum_demo/` (config + bundled synthetic-or-fixture snapshot), `README.md`, `tests/report/…`, `tests/test_end_to_end.py`.

**Interfaces:** Consumes everything. Produces `build_report`, `GateConfig`, the go/no-go verdict, and the demo pipeline.

**Acceptance criteria (mirrors the Definition of Done):**
- [ ] One command runs: data (fixture snapshot) → features/labels → triage sweep → event-driven backtest (net) → validation stats → survival MC → report artifact. Test executes the full chain offline.
- [ ] Report contains: gross vs net performance, CV results per scheme, DSR (with n_trials printed **from the TrialLogger**, plus the trial list hash), PBO, bootstrap CI, survival probability + CI, all red flags, config/data hashes, and a GO / NO-GO verdict strictly derived from `GateConfig` (survival ≥ 0.90, DSR significance, PBO ceiling, no fail-severity flags). Verdict logic unit-tested on crafted inputs both sides of each gate.
- [ ] Demo strategy is expected to produce **NO-GO** on the bundled fixture (honest demo; test asserts the pipeline is capable of saying no).
- [ ] README documents architecture (diagram), config surface, validation gates, prop presets, how to add a provider/strategy, and the yfinance NOT-FOR-VALIDATION policy.
- [ ] Repo-wide checks green: `grep` test asserting `yfinance` appears only under `data/adapters/yfinance_dev.py` and docs; mypy/ruff/pytest/CI all green.

---

## Review protocol (every task)

1. Subagent receives: this plan's Global Constraints + Shared Contracts + its own task section + paths to legacy reference files (read-only).
2. Deliverable: code + tests + self-report (built/assumptions/limitations).
3. Fable review checklist: acceptance criteria line-by-line; G1–G16 sweep; contract-drift check against Shared Contracts; test-quality check (tests must fail without the implementation — spot-verify); no unexplained dependencies.
4. Written feedback → revise → resubmit until **APPROVED**. Merge order follows waves; each wave merges only after all its tasks are approved.

## Definition of Done (Fable's final checklist)

- [ ] yfinance absent from all validation paths (repo-wide test); data layer point-in-time with continuous-contract metadata.
- [ ] End-to-end run works (T9 pipeline test).
- [ ] All kill switches unit-tested, incl. disconnect + stale-data.
- [ ] Trial counter proves every tested configuration is logged and feeds the DSR.
- [ ] README documents architecture, config, validation gates.
