# Design Spec — futures-engine Research Cockpit (local UI)

**Date:** 2026-07-30
**Status:** Approved design, pre-implementation
**Author:** brainstormed with the user (Fable-architect / Opus-implementer protocol continues)

## 1. Context & goal

`futures-engine` (MES/MNQ CME micro-futures research + execution system, master @ `42935b3`, 458 tests green) is currently a **CLI research pipeline**: `run_pipeline(...)` chains data → features/labels → triage sweep → Nautilus event-driven backtest → validation stats → prop-survival MC → a GO/NO-GO report artifact (`report.md`/`report.html`/`verdict.json`/`manifest.json`). There is no interactive UI, and the market-data providers are fixture-tested stubs (not wired to a live feed).

**Goal (this spec):** build a local **research cockpit** — a browser UI to configure and launch strategy sweeps against real or synthetic data, then browse the GO/NO-GO report, validation stats, survival, and red flags — and wire a **real Databento fetch path** so the cockpit can run on real MES/MNQ history.

## 2. Scope

**In scope (Phase 1 — this spec):**
- FastAPI + HTMX local web app (`futures_engine/ui/`) with five screens: Configure, Data, Run, Report, History.
- In-browser config builder that produces a validated `PipelineConfig` (no hand-editing YAML).
- Data panel: list existing snapshots; generate a synthetic snapshot; **fetch real MES/MNQ bars from Databento** into a content-addressed snapshot (behind a `DATABENTO_API_KEY`).
- Async run execution (CPU-bound sweep in a process pool) with HTMX progress polling.
- Rich report view + run-history browser, rendered from existing run artifacts.

**Out of scope (deferred / later phases):**
- **Phase 2 — live trading console** (connect a prop-firm account, stream positions / kill-switch state, operate live). Architected-for but not built: the cockpit stays firm-/provider-agnostic and reuses the existing `ExecutionClient` seam so the live console slots in later, once a strategy passes the gate and a firm is chosen.
- Decimal/fixed-point price storage (see §7 — we keep the existing float64 `Bars` schema).
- Validating the Databento adapter against Databento's *live* response shapes — requires an API key; documented fast-follow in §9.

## 3. Decisions (from brainstorming)

1. **Purpose:** research cockpit now; grows into a live console later (research must clear a GO verdict before live trading).
2. **Stack:** FastAPI + HTMX + server-rendered Jinja templates, no JS build step; local single-user, binds `127.0.0.1` only, no auth. Started via a `futures-cockpit` console script → `http://localhost:8000`.
3. **Thin-orchestration rule (non-negotiable):** the UI reimplements **no** trading / cost / validation / sizing / data logic. It calls the existing audited modules (`run_pipeline`, `SnapshotStore`, the synthetic generator, `build_continuous`, the Databento adapter, `build_report`) exactly as the CLI does. This keeps G1–G16 intact and prevents a second, unaudited code path.
4. **Real data now, Databento-first,** but exercised offline via fixtures/synthetic until a key is present; flips to live on `DATABENTO_API_KEY`.

## 4. Architecture & package layout

New package `futures_engine/ui/`:

```
futures_engine/ui/
  app.py          # FastAPI app factory, routes, 127.0.0.1 bind, console-script entry
  jobs.py         # in-process job registry + ProcessPoolExecutor for CPU-bound runs; thread pool for I/O fetch
  config_form.py  # form fields <-> PipelineConfig (validate via existing pydantic; unknown keys already rejected)
  data_panel.py   # list snapshots / generate synthetic / Databento fetch (behind key)
  reportview.py   # load a run's verdict.json/report.md/manifest.json -> template context
  templates/      # Jinja: base, cockpit(configure), data, run-progress, report, history
  static/         # vendored htmx.min.js (offline), minimal CSS
```

Console script in `pyproject.toml`: `futures-cockpit = "futures_engine.ui.app:main"`. New deps go in **optional extras, not core** (the audited core that feeds the pipeline stays lean): a **`ui` extra** (`fastapi`, `uvicorn`, `jinja2`, `python-multipart` for form posts) and a **`data` extra** (`databento`). Install with `pip install -e ".[dev,ui,data]"`. The `ui` package imports its web deps lazily so importing `futures_engine` without the extra never fails. HTMX is vendored as a static asset (no CDN — offline discipline).

## 5. Components — the five screens

1. **Configure** — form: instrument (MES/MNQ), strategy family + param grid, cost preset, prop preset (Topstep/MFFU/Apex), `GateConfig` thresholds (defaults 0.95 / 0.5 / 0.90). Submit → validates through the existing `PipelineConfig`/`Settings` pydantic models → writes the run config. Validation errors render inline; no silent coercion.
2. **Data** — table of snapshots in the store (hash, instrument, interval, range, validation-grade, ContinuousMeta present); "generate synthetic" panel (params → existing generator → validation-grade snapshot); "fetch real" panel (Databento) — **disabled with a clear note when no key is present**.
3. **Run** — `POST /runs` enqueues a job (`job_id`) running `run_pipeline` in a **ProcessPoolExecutor**; a progress line polls `GET /runs/{id}/status` via HTMX `hx-trigger="every 2s"`, stopping on terminal state; on success → redirect to the report.
4. **Report** — verdict banner (GO/NO-GO) with each gate's pass/fail + numbers, equity curve, gross-vs-net, CV/DSR/PBO/bootstrap-CI, survival + CI, red flags, config/data/git hashes. Rendered from the run's existing `verdict.json` + `report.md` — nothing recomputed.
5. **History** — table of past runs from `out/` (timestamp, strategy, verdict, key metrics), each linking to its report.

## 6. Data flow

**Fetch (real data):** Data panel → `POST /data/fetch` → Databento call in a **thread pool** (I/O-bound) via `databento.Historical(...).timeseries.get_range(dataset="GLBX.MDP3", symbols=…, schema="ohlcv-1m", start, end, stype_in=…)` → `DBNStore.to_df()` → per-contract `Bars` → existing `build_continuous` → `SnapshotStore.save` (content hash, `ContinuousMeta`, `validation_grade=True`). Snapshot only on a clean, complete fetch.

**Run (sweep + verdict):** Configure → validated `PipelineConfig` → `POST /runs` → job runs `run_pipeline` in a **ProcessPoolExecutor** (CPU-bound pandas; must not block the event loop or a GIL-bound thread) → status tracked in `jobs.py` registry → HTMX polls → report reads the written artifacts.

## 7. External-dependency facts (grounded by research 2026-07-30)

**Databento SDK:**
- Client `databento.Historical('KEY')`, or key omitted → falls back to `DATABENTO_API_KEY` env var.
- `client.timeseries.get_range(dataset, symbols, schema, start, end, stype_in=..., stype_out=..., limit=...)` → `DBNStore`; `.to_df()` → pandas.
- Dataset **`GLBX.MDP3`**; 1-min OHLCV schema **`ohlcv-1m`**.
- **Continuous-contract decision:** default to **parent symbology** (`MES.FUT` / `MNQ.FUT`, `stype_in="parent"`) → individual contracts → **our `build_continuous`** (preserves our roll rule + `ContinuousMeta`, satisfies G3). Offer Databento's own continuous symbols (`MES.c.0`, `stype_in="continuous"`) as a clearly-labeled "trust-their-roll, quick" alternative. Nautilus's own Databento adapter does not build continuity, so we fetch via the raw SDK — consistent with our data layer being the continuity authority.
- **Prices:** DBN encodes prices as fixed-point int×1e-9, but `.to_df()` returns floats by default (`price_type="float"`). We keep the existing **float64 `Bars` schema** (used throughout the audited stack and the T1 content hash) — fetch with the default float representation; MES/MNQ tick magnitudes (~5000, tick 0.25) are well within float64 precision. `price_type="decimal"` is available if exact arithmetic is ever required; noted, not adopted (avoids a disruptive change to the audited snapshot schema).
- **Index `ts_event` = bar-open** time (UTC ns) — matches our Bars index / next-bar-open causality; the adapter preserves this and must not relabel it as bar-close.
- **Verify-on-key:** exact `get_range` keyword defaults (e.g. `limit`, `stype_out`) were not pulled from the doc page during research; confirm via `help(get_range)` + one real pull when the key is added (see §9).

**FastAPI + HTMX:**
- Job pattern: `POST /runs` → `job_id` → HTMX `hx-trigger="every 2s"` polling a status fragment, stopping on terminal state.
- **CPU-bound trap (confirmed):** pandas/numpy hold the GIL, so a thread pool does **not** free the event loop; the sweep runs in a `ProcessPoolExecutor` (or subprocess). The Databento HTTP fetch is I/O-bound and may use a thread pool.

## 8. Error handling & safety

- Local bind `127.0.0.1` only; single-user; no auth.
- Config errors → existing pydantic messages inline on the form; no silent coercion.
- Databento errors (bad key, rate limit, empty range) → clear panel error; **never a partial snapshot** (`save` only on clean fetch).
- Crashed pipeline job → registry marks it `failed`, traceback shown on the run page, artifacts absent.
- `DATABENTO_API_KEY` read from env only; never logged; never written into a run's config or manifest. Only a boolean "real-data provider used" + the snapshot hash are recorded, preserving reproducibility provenance without leaking the secret.

## 9. Testing

- FastAPI `TestClient`, fully offline (G15 — no network in tests).
- Route rendering; config round-trip form → `PipelineConfig` → YAML; validation-error rendering.
- Synthetic **end-to-end**: `POST /runs` → job → report page asserts the NO-GO verdict renders correctly.
- Databento adapter against **recorded DBN fixtures**: fixed-point→float mapping correct; `ts_event` mapped to bar-open index; parent-symbology bars flow through `build_continuous`.
- `jobs.py` registry transitions (queued → running → done/failed) unit-tested; ProcessPoolExecutor path exercised with a trivial job.
- **Fast-follow requiring a key (documented, not in this build):** confirm `help(get_range)` signature + one real MES pull to validate live response shape against the fixtures.

## 10. Prerequisites & open items

- A **Databento API key** is required to exercise the live fetch (env `DATABENTO_API_KEY`); until then the panel is disabled and the cockpit runs on synthetic/existing snapshots.
- Phase-2 live console (Tradovate `isAutomated` field is concretely documented; TopstepX/ProjectX auto-order tagging is thinner and unverified) is a separate future spec.
