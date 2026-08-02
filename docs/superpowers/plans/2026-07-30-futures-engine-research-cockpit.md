# Research Cockpit Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. **Orchestration protocol (same as the T0–T9 build):** Fable (architect) assigns/reviews; Opus subagents write all code and tests; iterate to APPROVED per task; worktree isolation; merges + pushes by Fable. Design spec: `docs/superpowers/specs/2026-07-30-futures-engine-research-cockpit-design.md`.

**Goal:** A local browser "research cockpit" (FastAPI + HTMX) to configure and launch strategy sweeps against synthetic or real (Databento) data and browse the GO/NO-GO report — a thin UI over the existing audited pipeline.

**Architecture:** New `futures_engine/ui/` package. The UI reimplements **no** trading/cost/validation/sizing/data logic — it orchestrates the existing modules (`run_pipeline`, `SnapshotStore`, the synthetic generator, `build_continuous`, `DatabentoAdapter`, the report artifacts). CPU-bound sweeps run in a `ProcessPoolExecutor`; the Databento HTTP fetch runs in a thread pool. Binds `127.0.0.1` only.

**Tech Stack:** FastAPI, Uvicorn, Jinja2, python-multipart, HTMX (vendored, offline), the existing `databento` SDK adapter. Python 3.12+ (installed 3.14), pandas 3.0, pydantic v2.

## Global Constraints (binding; violations = review rejection)

- **UI-G1 Thin orchestration:** the UI calls existing audited functions; it must NOT re-implement pipeline, cost, validation, sizing, continuous-contract, or report logic. No second code path for anything under G1–G16.
- **UI-G2 Optional extras, lazy import:** UI/data deps live in `[project.optional-dependencies]` extras `ui` (`fastapi`, `uvicorn`, `jinja2`, `python-multipart`) and `data` (`databento`) — NOT core. `import futures_engine` (and the existing 458-test suite) must still pass with neither extra installed; `futures_engine.ui.*` imports its web deps lazily.
- **UI-G3 Offline tests (inherits G15):** every test runs offline — FastAPI `TestClient`, recorded DBN fixtures, synthetic snapshots. No network in CI. HTMX is vendored as a static asset (no CDN).
- **UI-G4 Local-only + secret hygiene:** server binds `127.0.0.1`; `DATABENTO_API_KEY` is read from env only, never logged, never written into any run config/manifest/report (only a boolean "real-data used" + the snapshot hash are recorded).
- **UI-G5 CPU-bound isolation:** `run_pipeline` executes in a `ProcessPoolExecutor` (or subprocess), never on the event loop or a thread pool (pandas holds the GIL). Databento HTTP fetch (I/O-bound) may use a thread pool.
- **UI-G6 float64 Bars schema preserved:** Databento bars enter as float64 (the adapter's existing `price_type` default), matching the audited `Bars`/`SnapshotStore` schema; do not introduce a decimal/fixed representation.
- **UI-G7 Reproducibility preserved:** the UI builds a `PipelineConfig` and calls `run_pipeline` exactly as the CLI does; it adds no new provenance path. `ts_event` from Databento is the bar-OPEN timestamp — the adapter/continuous path must preserve that (never relabel as close).
- **UI-G8 Typed/clean:** mypy + ruff + `ruff format --check` clean; every route/handler typed; tests pristine (no warnings).

## Shared Contracts (consume from master — do NOT redefine)

```text
futures_engine.pipeline.run:
  PipelineConfig(run_id, instrument, signal, strategy_family, grid: dict[str,list[int]],
                 snapshot_hash, prop_preset, seed=7, n_splits=5, embargo_frac=0.02,
                 pbo_partitions=8, bootstrap_n=500, survival: SurvivalSettings, gates: GateConfig)
    .load(path) -> PipelineConfig     # model_validate(load_yaml), extra keys rejected
  SurvivalSettings(n_paths>0, horizon_days>0, contracts>=0)
  run_pipeline(config_path, *, settings_dir, snapshot_root, out_dir,
               created_at=None, feature_config=None, repo=None) -> PipelineOutcome
  PipelineOutcome(run_id, run_dir: Path, verdict: Verdict, n_trials: int, best_params: dict); .decision -> str
futures_engine.report.builder: GateConfig(min_dsr_p=0.95, max_pbo=0.5, min_survival=0.90)
futures_engine.backtest.strategy_adapter: SIGNAL_REGISTRY = {"donchian_breakout": ..., "ma_cross_vol_target": ...}
futures_engine.data.store: SnapshotStore(root).save(bars, meta)->hash ; .load(hash)->(Bars, DatasetMeta)
futures_engine.data.continuous: build_continuous(per_contract, specs, roll_rule, adjustment, *, ...) -> (Bars, ContinuousMeta)
futures_engine.data.adapters.databento_adapter: DatabentoAdapter(api_key).fetch_bars(...) / .list_contracts(symbol_root, start, end)->list[ContractInfo]
futures_engine.core.config: Settings loader (instruments/costs/prop_rules YAML)
Prop presets (configs/prop_rules.yaml): topstep_50k, mffu_50k, apex_50k
Run artifacts per run dir: report.md, report.html, verdict.json {decision, gates:[{name,passed,detail}]}, manifest.json
```

---

### Task U1: UI scaffold — app factory, extras, base template, smoke test

**Files:** Create `futures_engine/ui/{__init__,app.py}`, `futures_engine/ui/templates/{base.html,index.html}`, `futures_engine/ui/static/{htmx.min.js,cockpit.css}`, `tests/ui/{__init__,conftest.py,test_app.py}`. Modify `pyproject.toml` (`[project.optional-dependencies]` `ui`/`data` extras; `[project.scripts]` `futures-cockpit`).

**Interfaces:**
- Consumes: nothing from other UI tasks.
- Produces: `futures_engine.ui.app.create_app() -> FastAPI` (app factory); `futures_engine.ui.app.main() -> None` (console entry: `uvicorn.run(create_app(), host="127.0.0.1", port=8000)`); a Jinja `templates` env + `static` mount; base layout with a nav (Configure / Data / Runs) and the vendored `htmx.min.js`.

**Acceptance criteria:**
- [ ] `pip install -e ".[dev,ui,data]"` installs the extras; `import futures_engine` and the full existing suite still pass with NO extras installed (UI-G2 — test the lazy-import boundary: `futures_engine.ui.app` importing without fastapi raises a clear message, but `import futures_engine` never does).
- [ ] `create_app()` returns a FastAPI app bound conceptually to `127.0.0.1`; `GET /` renders the index (nav + HTMX loaded from `/static/htmx.min.js`, not a CDN — UI-G3).
- [ ] `TestClient(create_app()).get("/")` → 200 and contains the nav links; `/static/htmx.min.js` → 200.
- [ ] mypy/ruff/format clean; `main()` exists and is wired as the `futures-cockpit` script (do not invoke a live server in tests).

---

### Task U2: Job registry + process-pool run execution (`jobs.py`)

**Files:** Create `futures_engine/ui/jobs.py`, `tests/ui/test_jobs.py`.

**Interfaces:**
- Consumes: nothing (generic job runner; the run wiring is U5).
- Produces:
  `JobStatus = Literal["queued","running","done","failed"]`;
  `JobRecord(id: str, status: JobStatus, result: object | None, error: str | None, started_at, finished_at)`;
  `class JobRegistry: submit(fn: Callable[[], T], *, job_id: str | None=None) -> str` (runs `fn` in a `ProcessPoolExecutor`, tracks status), `get(job_id) -> JobRecord`, `list() -> list[JobRecord]`. A module-level singleton accessor `get_registry() -> JobRegistry`.

**Acceptance criteria:**
- [ ] A submitted trivial job transitions queued → running → **done** and exposes its return value; a job whose `fn` raises transitions to **failed** with the traceback string captured in `error` (not swallowed) — both tested.
- [ ] Execution uses a `ProcessPoolExecutor` (UI-G5); the submitted callable must be top-level/picklable — the test uses a module-level function and asserts the pool path (not a thread). A picklability failure surfaces as a `failed` job with a clear error, not a hang.
- [ ] `get()` on an unknown id raises `KeyError`; `list()` returns records newest-first. Deterministic/offline; no wall-clock assertions beyond ordering. mypy/ruff/format clean.

---

### Task U3: Data panel — list snapshots + generate synthetic (`data_panel.py`, screens)

**Files:** Create `futures_engine/ui/data_panel.py`, `futures_engine/ui/templates/data.html`, `tests/ui/test_data_panel.py`. Modify `futures_engine/ui/app.py` (routes).

**Interfaces:**
- Consumes: `SnapshotStore`, the synthetic MES generator (reuse `examples/mes_momentum_demo/generate_snapshot.py` logic or the `tests/*/conftest.py` generator — import/refactor into a reusable `futures_engine.data.synthetic` helper IF one does not exist; otherwise call the existing generator). `JobRegistry` (U2) NOT required here (generation is fast/synchronous).
- Produces: `list_snapshots(store_root) -> list[SnapshotSummary]` where `SnapshotSummary(hash, symbol_root, interval, start, end, validation_grade, has_continuous_meta)`; `generate_synthetic(store_root, *, symbol_root, n_bars, seed, ...) -> str` (returns snapshot hash; writes a validation-grade snapshot WITH `ContinuousMeta`). Routes: `GET /data` (panel), `POST /data/synthetic` (generate → redirect/refresh list).

**Acceptance criteria:**
- [ ] `GET /data` lists snapshots in the store with the summary fields; empty store renders an empty-state, not an error.
- [ ] `POST /data/synthetic` creates a **validation-grade** snapshot with `ContinuousMeta` (loadable by the pipeline; `require_validation_grade` passes) — asserted by loading it back and by it subsequently being runnable in U5's test.
- [ ] The "fetch real (Databento)" section renders **disabled with a clear note** when `DATABENTO_API_KEY` is absent (the live path is U7). No network. mypy/ruff/format clean.

---

### Task U4: Configure screen — config builder → validated `PipelineConfig`

**Files:** Create `futures_engine/ui/config_form.py`, `futures_engine/ui/templates/configure.html`, `tests/ui/test_config_form.py`. Modify `futures_engine/ui/app.py` (routes).

**Interfaces:**
- Consumes: `PipelineConfig`, `SurvivalSettings`, `GateConfig`, `SIGNAL_REGISTRY` (signal choices), prop presets from `configs/prop_rules.yaml`, snapshot hashes from U3's `list_snapshots`.
- Produces: `form_to_config(form: Mapping[str,str]) -> PipelineConfig` (parse + validate via the existing pydantic model; grid parsed from a text field into `dict[str,list[int]]`); `config_to_yaml(cfg, path)` writer. Routes: `GET /configure` (form pre-filled with signal/preset/snapshot choices), `POST /configure` (validate → persist config → hand off to Run, or re-render with inline errors).

**Acceptance criteria:**
- [ ] A well-formed form round-trips `form_to_config(...)` → `PipelineConfig` with the exact field values (signal ∈ SIGNAL_REGISTRY, prop_preset ∈ presets, gates default 0.95/0.5/0.90 unless overridden, grid parsed to `dict[str,list[int]]`).
- [ ] Invalid input (unknown signal, empty grid, malformed grid text, out-of-range gate) re-renders the form with the **existing pydantic error messages inline** — no silent coercion, no 500 (tested for ≥3 distinct invalid cases).
- [ ] `strategy_family` is auto-derived from the chosen signal's family (NOT a free-text field the user can mismatch) — preserves the honest-DSR guarantee from the T9 fix. mypy/ruff/format clean.

---

### Task U5: Run + Report — config → job → `run_pipeline` → report view

**Files:** Create `futures_engine/ui/reportview.py`, `futures_engine/ui/templates/{run_progress.html,report.html}`, `tests/ui/test_run_report.py`. Modify `futures_engine/ui/app.py` (routes).

**Interfaces:**
- Consumes: `JobRegistry` (U2), `form_to_config`/`config_to_yaml` (U4), `run_pipeline`, `PipelineOutcome`, a run's artifacts (`verdict.json`, `report.md`, `manifest.json`).
- Produces: `submit_run(cfg_path, *, settings_dir, snapshot_root, out_dir) -> str` (job_id; the job calls `run_pipeline` in the process pool); `load_report(run_dir) -> ReportContext` (parses verdict.json + report.md + manifest.json into template context — NO recomputation). Routes: `POST /runs` (enqueue → progress page), `GET /runs/{id}/status` (HTMX fragment; `hx-trigger="every 2s"`, stops on terminal), `GET /runs/{id}/report` (report page).

**Acceptance criteria:**
- [ ] **End-to-end offline:** `POST /runs` with a synthetic snapshot (from U3) enqueues a job that runs the real `run_pipeline`; the status fragment reports running → done; `GET /runs/{id}/report` renders the verdict banner with each gate's pass/fail + numbers, and shows the **NO-GO** demo-style result. Numbers come from `verdict.json`/`report.md` — nothing recomputed (UI-G1).
- [ ] The progress fragment stops polling on terminal state (asserted: terminal fragment has no `hx-trigger`); a **failed** run renders the traceback on the run page with artifacts absent (not a 500).
- [ ] The report page embeds the run's `report.html` content (or renders from `report.md`) plus config/data/git hashes from `manifest.json`. mypy/ruff/format clean. (This is the integration task; keep the run itself in the process pool per UI-G5.)

---

### Task U6: History screen

**Files:** Create `futures_engine/ui/templates/history.html`, `tests/ui/test_history.py`. Modify `futures_engine/ui/{app.py,reportview.py}`.

**Interfaces:**
- Consumes: the `out/` run directories, `load_report`/a lighter `summarize_run(run_dir) -> RunSummary(run_id, created_at, signal, decision, key_metrics)`.
- Produces: `list_runs(out_dir) -> list[RunSummary]` (newest-first, tolerant of partial/failed run dirs); Route `GET /history`.

**Acceptance criteria:**
- [ ] `GET /history` lists completed runs newest-first with run_id, strategy/signal, GO/NO-GO decision, and a couple of key metrics, each row linking to `/runs/{id}/report`.
- [ ] A partial/failed run dir (missing `verdict.json`) is listed as `failed`/`incomplete` rather than crashing the page (tested). Empty `out/` → empty-state. mypy/ruff/format clean.

---

### Task U7: Databento real-data fetch (wire existing adapter + fixtures)

**Files:** Modify `futures_engine/ui/data_panel.py`, `futures_engine/ui/templates/data.html`, `futures_engine/ui/app.py`. Create `tests/ui/test_databento_fetch.py`, `tests/ui/fixtures/` (recorded DBN → DataFrame fixture, offline). Possibly add `futures_engine/data/adapters/databento_adapter.py` validation-only helpers (do NOT change its audited fetch logic).

**Interfaces:**
- Consumes: existing `DatabentoAdapter(api_key)` (`.fetch_bars`, `.list_contracts`), `build_continuous`, `SnapshotStore`, `JobRegistry` (U2 — fetch may be long; run the HTTP fetch in a thread pool inside a job).
- Produces: `fetch_databento_snapshot(store_root, *, symbol_root, start, end, roll_rule, adjustment, api_key) -> str` (parent-symbology individual contracts via the adapter → `build_continuous` → validation-grade snapshot with `ContinuousMeta`; returns hash). Route `POST /data/fetch` (enabled only when a key is present).

**Acceptance criteria:**
- [ ] With `DATABENTO_API_KEY` present the panel enables the fetch form (dataset `GLBX.MDP3`, schema `ohlcv-1m`, MES/MNQ, date range, roll rule + adjustment). Default symbology = **parent** (individual contracts → our `build_continuous`, preserving `ContinuousMeta`/G3); Databento's own continuous (`MES.c.0`, `stype_in="continuous"`) offered as a labeled "trust-their-roll" alternative.
- [ ] **Offline fixture test (UI-G3):** feed a recorded DBN/`to_df()` fixture through the adapter mapping → assert fixed-point→float64 prices (×1e-9) correct, `ts_event` mapped to the bar-**open** index (UI-G6/G7), columns `open/high/low/close/volume`; then through `build_continuous` → a validation-grade snapshot with `ContinuousMeta`. No network.
- [ ] A fetch error (bad key / empty range / HTTP error) surfaces a clear panel error and writes **no partial snapshot** (`save` only on a clean, complete fetch) — tested with a stubbed failing adapter. The API key is never logged nor written into the snapshot meta/manifest (UI-G4) — assert the meta/manifest contain only the hash + a "real-data" boolean.
- [ ] **Documented (not automated) fast-follow:** validating against Databento's *live* response requires a key — a one-line note in the report/panel that `help(get_range)` + one real MES pull should confirm the live shape before trusting live fetches. mypy/ruff/format clean.

---

## Definition of Done

- [ ] `futures-cockpit` launches a local app on `127.0.0.1:8000`; Configure → Data → Run → Report → History all work on synthetic data offline.
- [ ] With a `DATABENTO_API_KEY`, the Data panel fetches real MES/MNQ history into a validation-grade snapshot; without one, the cockpit is fully usable on synthetic/existing snapshots.
- [ ] `import futures_engine` + the pre-existing 458-test suite pass with NO ui/data extras installed; the new UI suite passes with the extras. mypy/ruff/format clean across the repo.
- [ ] No trading/cost/validation/sizing/data/report logic reimplemented in `ui/` (UI-G1); `DATABENTO_API_KEY` never logged or persisted (UI-G4); runs execute in a process pool (UI-G5).
