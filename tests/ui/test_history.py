"""History screen tests for the cockpit (U6).

The History screen lists past runs newest-first, each row linking to its report.
It is tolerant of partial/failed run dirs (missing ``verdict.json``) -- such a run
is listed as ``incomplete`` rather than crashing the page (UI-G1). An empty runs
root yields an empty-state, not an error. Everything reads on-disk artifacts only
and recomputes nothing; fixtures are tiny hand-written files (no real pipeline).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from fastapi.testclient import TestClient

from futures_engine.ui.app import create_app
from futures_engine.ui.reportview import RunSummary, list_runs, summarize_run

_CONFIGS = Path(__file__).resolve().parents[2] / "configs"


def _write_run(
    runs_root: Path,
    run_id: str,
    *,
    signal: str = "donchian_breakout",
    decision: str | None = "NO-GO",
    mtime: float | None = None,
) -> Path:
    """Write a minimal run dir; omit ``verdict.json`` when ``decision`` is None."""
    run_dir = runs_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.yaml").write_text(
        f"run_id: {run_id}\nsignal: {signal}\nstrategy_family: trend_momentum\n",
        encoding="utf-8",
    )
    if decision is not None:
        (run_dir / "verdict.json").write_text(
            json.dumps(
                {
                    "decision": decision,
                    "gates": [
                        {"name": "survival", "passed": True, "detail": "p_survival=0.9500"},
                        {"name": "pbo", "passed": False, "detail": "PBO=0.6000"},
                    ],
                    "fail_flag_codes": [],
                }
            ),
            encoding="utf-8",
        )
        (run_dir / "report.html").write_text("<html>report</html>", encoding="utf-8")
    if mtime is not None:
        os.utime(run_dir, (mtime, mtime))
    return run_dir


# --- summarize_run ----------------------------------------------------------


def test_summarize_run_complete(tmp_path: Path) -> None:
    run_dir = _write_run(tmp_path, "alpha", signal="ema_cross", decision="GO")
    summary = summarize_run(run_dir)
    assert isinstance(summary, RunSummary)
    assert summary.run_id == "alpha"
    assert summary.signal == "ema_cross"
    assert summary.decision == "GO"
    assert summary.key_metrics  # a couple of gate details
    assert summary.created_at is not None


def test_summarize_run_missing_verdict_is_incomplete(tmp_path: Path) -> None:
    run_dir = _write_run(tmp_path, "partial", decision=None)
    summary = summarize_run(run_dir)
    assert summary.run_id == "partial"
    assert summary.signal == "donchian_breakout"  # still read from config.yaml
    assert summary.decision == "incomplete"
    assert summary.key_metrics == {}


def test_summarize_run_missing_config_signal_unknown(tmp_path: Path) -> None:
    run_dir = tmp_path / "bare"
    run_dir.mkdir()
    summary = summarize_run(run_dir)
    assert summary.signal == "?"
    assert summary.decision == "incomplete"


# --- list_runs --------------------------------------------------------------


def test_list_runs_newest_first(tmp_path: Path) -> None:
    _write_run(tmp_path, "oldest", mtime=1_000.0)
    _write_run(tmp_path, "middle", mtime=2_000.0)
    _write_run(tmp_path, "newest", mtime=3_000.0)
    summaries = list_runs(tmp_path)
    assert [s.run_id for s in summaries] == ["newest", "middle", "oldest"]


def test_list_runs_empty_root(tmp_path: Path) -> None:
    assert list_runs(tmp_path / "does_not_exist") == []
    assert list_runs(tmp_path) == []


def test_list_runs_tolerates_partial_dir(tmp_path: Path) -> None:
    _write_run(tmp_path, "good", decision="GO", mtime=1_000.0)
    _write_run(tmp_path, "failed", decision=None, mtime=2_000.0)
    summaries = list_runs(tmp_path)
    assert [s.run_id for s in summaries] == ["failed", "good"]
    by_id = {s.run_id: s for s in summaries}
    assert by_id["failed"].decision == "incomplete"
    assert by_id["good"].decision == "GO"


def test_list_runs_bad_dir_does_not_kill_list(tmp_path: Path) -> None:
    _write_run(tmp_path, "good", decision="GO")
    # A run dir whose verdict.json is corrupt must not sink the whole list.
    bad = tmp_path / "corrupt"
    bad.mkdir()
    (bad / "config.yaml").write_text("run_id: corrupt\nsignal: x\n", encoding="utf-8")
    (bad / "verdict.json").write_text("{not valid json", encoding="utf-8")
    summaries = list_runs(tmp_path)
    run_ids = {s.run_id for s in summaries}
    assert "good" in run_ids
    # corrupt is either listed as incomplete or skipped, but never raises.
    assert "corrupt" not in run_ids or by_decision(summaries, "corrupt") == "incomplete"


def by_decision(summaries: list[RunSummary], run_id: str) -> str | None:
    for s in summaries:
        if s.run_id == run_id:
            return s.decision
    return None


# --- GET /history route -----------------------------------------------------


def test_history_route_lists_runs_newest_first(tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    _write_run(runs_root, "run_old", decision="GO", mtime=1_000.0)
    _write_run(runs_root, "run_new", decision="NO-GO", mtime=2_000.0)
    app = create_app(runs_root=runs_root, configs_dir=_CONFIGS)
    client = TestClient(app)

    resp = client.get("/history")
    assert resp.status_code == 200
    body = resp.text
    assert "run_new" in body and "run_old" in body
    assert body.index("run_new") < body.index("run_old")  # newest first
    # Each row links to the run's report.
    assert "/runs/run_new/report" in body
    assert "/runs/run_old/report" in body


def test_history_route_failed_run_does_not_crash(tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    _write_run(runs_root, "doomed", decision=None)
    app = create_app(runs_root=runs_root, configs_dir=_CONFIGS)
    client = TestClient(app)

    resp = client.get("/history")
    assert resp.status_code == 200
    assert "doomed" in resp.text
    assert "incomplete" in resp.text.lower()


def test_history_route_empty_state(tmp_path: Path) -> None:
    app = create_app(runs_root=tmp_path / "empty", configs_dir=_CONFIGS)
    client = TestClient(app)
    resp = client.get("/history")
    assert resp.status_code == 200
    assert "no runs" in resp.text.lower()


def test_history_in_nav(tmp_path: Path) -> None:
    app = create_app(runs_root=tmp_path, configs_dir=_CONFIGS)
    client = TestClient(app)
    resp = client.get("/history")
    assert 'href="/history"' in resp.text
