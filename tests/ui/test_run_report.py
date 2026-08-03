"""End-to-end run + report tests for the cockpit (U5).

The load-bearing test runs the REAL ``run_pipeline`` in the U2 process pool on a
tiny synthetic snapshot, polls the HTMX status fragment to terminal, and asserts
the NO-GO verdict and its gate rows render from ``verdict.json`` -- nothing is
recomputed (UI-G1). Everything is offline and kept small so it stays fast.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from futures_engine.pipeline.run import PipelineConfig
from futures_engine.ui.app import create_app
from futures_engine.ui.config_form import config_to_yaml
from futures_engine.ui.data_panel import generate_synthetic
from futures_engine.ui.jobs import JobRegistry
from futures_engine.ui.reportview import _run_pipeline_job, load_report, submit_run

_REPO = Path(__file__).resolve().parents[2]
_CONFIGS = _REPO / "configs"

_FAKE_HASH = "de" * 32  # 64 hex chars, not present in any store


@pytest.fixture
def registry() -> Iterator[JobRegistry]:
    reg = JobRegistry()
    try:
        yield reg
    finally:
        reg.shutdown()


def _write_config(
    runs_root: Path,
    snapshot_root: Path,
    *,
    run_id: str = "synthetic_nogo",
    snapshot_hash: str | None = None,
) -> Path:
    """Write a tiny-but-valid PipelineConfig to ``runs_root/<run_id>/config.yaml``."""
    if snapshot_hash is None:
        snapshot_hash = generate_synthetic(snapshot_root, symbol_root="MES", n_bars=400, seed=7)
    cfg = PipelineConfig.model_validate(
        {
            "run_id": run_id,
            "instrument": "MES",
            "signal": "donchian_breakout",
            "strategy_family": "trend_momentum",
            "grid": {"window": [5, 10]},
            "snapshot_hash": snapshot_hash,
            "prop_preset": "topstep_50k",
            "seed": 7,
            "n_splits": 3,
            "embargo_frac": 0.02,
            "pbo_partitions": 4,
            "bootstrap_n": 50,
            "survival": {"n_paths": 50, "horizon_days": 30, "contracts": 1},
        }
    )
    path = runs_root / run_id / "config.yaml"
    config_to_yaml(cfg, path)
    return path


def _poll_to_terminal(client: TestClient, run_id: str, *, timeout: float = 180.0) -> str:
    """Poll the status fragment until it stops self-refreshing; return the fragment."""
    deadline = time.time() + timeout
    fragment = ""
    while time.time() < deadline:
        fragment = str(client.get(f"/runs/{run_id}/status").text)
        if "hx-trigger" not in fragment:  # terminal fragment drops the poller
            return fragment
        time.sleep(0.5)
    pytest.fail(f"run {run_id!r} did not terminate within {timeout}s")


# --- end-to-end NO-GO -------------------------------------------------------


def test_end_to_end_nogo_run(tmp_path: Path, registry: JobRegistry) -> None:
    snapshot_root = tmp_path / "snapshots"
    runs_root = tmp_path / "runs"
    _write_config(runs_root, snapshot_root)
    app = create_app(
        snapshot_root=snapshot_root,
        runs_root=runs_root,
        configs_dir=_CONFIGS,
        registry=registry,
    )
    client = TestClient(app)

    resp = client.post("/runs", data={"run_id": "synthetic_nogo"}, follow_redirects=False)
    assert resp.status_code == 303
    assert resp.headers["location"] == "/runs/synthetic_nogo"

    # A non-terminal fragment self-refreshes (contains hx-trigger).
    first = client.get("/runs/synthetic_nogo/status").text
    assert "hx-trigger" in first

    terminal = _poll_to_terminal(client, "synthetic_nogo")
    # Terminal fragment must NOT self-refresh (polling stops).
    assert "hx-trigger" not in terminal

    report = client.get("/runs/synthetic_nogo/report")
    assert report.status_code == 200
    # The honest synthetic run yields NO-GO; numbers come from verdict.json.
    assert "NO-GO" in report.text
    for gate in ("survival", "deflated_sharpe", "pbo", "no_fail_flags"):
        assert gate in report.text


# --- failed run: 200 with traceback, not a 500 -----------------------------


def test_failed_run_renders_traceback_not_500(tmp_path: Path, registry: JobRegistry) -> None:
    snapshot_root = tmp_path / "snapshots"
    runs_root = tmp_path / "runs"
    # A config whose snapshot_hash is absent -> run_pipeline raises in the worker.
    _write_config(runs_root, snapshot_root, run_id="doomed", snapshot_hash=_FAKE_HASH)
    app = create_app(
        snapshot_root=snapshot_root,
        runs_root=runs_root,
        configs_dir=_CONFIGS,
        registry=registry,
    )
    client = TestClient(app)

    client.post("/runs", data={"run_id": "doomed"}, follow_redirects=False)
    terminal = _poll_to_terminal(client, "doomed")
    assert "hx-trigger" not in terminal
    assert "failed" in terminal.lower()

    page = client.get("/runs/doomed")
    assert page.status_code == 200  # never a 500
    assert "no artifacts" in page.text.lower()

    # The report view degrades gracefully (no artifacts) rather than crashing.
    report = client.get("/runs/doomed/report")
    assert report.status_code == 200


# --- load_report (UI-G1: reads artifacts, recomputes nothing) --------------


def test_load_report_parses_artifacts(tmp_path: Path) -> None:
    snapshot_root = tmp_path / "snapshots"
    runs_root = tmp_path / "runs"
    cfg_path = _write_config(runs_root, snapshot_root, run_id="parsed")
    # Run synchronously (no pool) to produce artifacts, then read them back.
    _run_pipeline_job(str(cfg_path), str(_CONFIGS), str(snapshot_root), str(runs_root))

    ctx = load_report(runs_root / "parsed")
    assert ctx.complete is True
    assert ctx.decision == "NO-GO"
    gate_names = {g.name for g in ctx.gates}
    assert gate_names == {"survival", "deflated_sharpe", "pbo", "no_fail_flags"}
    assert ctx.config_hash  # from manifest.json
    assert ctx.git_sha
    assert ctx.report_html is not None and "Verdict" in ctx.report_html


def test_load_report_missing_artifacts_is_incomplete(tmp_path: Path) -> None:
    ctx = load_report(tmp_path / "never_ran")
    assert ctx.complete is False
    assert ctx.decision is None
    assert ctx.gates == []
    assert "verdict.json" in ctx.missing


# --- submit_run: job id == run id (UI-G5 picklable pool callable) -----------


def test_submit_run_job_id_is_run_id(tmp_path: Path, registry: JobRegistry) -> None:
    snapshot_root = tmp_path / "snapshots"
    runs_root = tmp_path / "runs"
    cfg_path = _write_config(runs_root, snapshot_root, run_id="named")
    job_id = submit_run(
        cfg_path,
        settings_dir=_CONFIGS,
        snapshot_root=snapshot_root,
        out_dir=runs_root,
        registry=registry,
    )
    # The job id IS the config run_id, so the run is addressable by one identifier.
    assert job_id == "named"
    assert registry.get("named").status in {"queued", "running", "done"}
