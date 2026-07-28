"""End-to-end acceptance: one command runs the whole research chain offline and
the honest demo yields NO-GO (the pipeline is capable of saying no).

The chain is: bundled validation-grade snapshot -> features/labels -> triage sweep
-> event-driven backtest (net) -> validation stats -> survival MC -> report artifact.
No network, no live execution adapters.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from futures_engine.pipeline.run import run_pipeline

_REPO = Path(__file__).resolve().parents[1]
_DEMO = _REPO / "examples" / "mes_momentum_demo"
_CONFIGS = _REPO / "configs"
_SNAPSHOTS = _DEMO / "snapshots"


@pytest.fixture(scope="module")
def outcome(tmp_path_factory: pytest.TempPathFactory):  # type: ignore[no-untyped-def]
    out_dir = tmp_path_factory.mktemp("reports")
    return run_pipeline(
        _DEMO / "config.yaml",
        settings_dir=_CONFIGS,
        snapshot_root=_SNAPSHOTS,
        out_dir=out_dir,
        repo=_REPO,
    )


def test_demo_is_no_go(outcome) -> None:  # type: ignore[no-untyped-def]
    # The honest demo MUST fail the verdict -- proving the system can say no.
    assert outcome.decision == "NO-GO"
    assert outcome.verdict.is_go is False
    # At least one gate must have actively failed.
    assert any(not g.passed for g in outcome.verdict.gates)


def test_report_artifacts_written(outcome) -> None:  # type: ignore[no-untyped-def]
    run_dir: Path = outcome.run_dir
    for name in ("report.md", "report.html", "verdict.json", "manifest.json"):
        assert (run_dir / name).exists(), name
    md = (run_dir / "report.md").read_text(encoding="utf-8")
    # Mandatory report contents (G10): gross vs net, DSR + honest n_trials +
    # trial hash, PBO, bootstrap CI, survival + CI, red flags, hashes, verdict.
    for token in (
        "Verdict: NO-GO",
        "gross vs net",
        "Deflated Sharpe",
        "n_trials (from TrialLogger)",
        "trial-list hash",
        "PBO",
        "Bootstrap Sharpe",
        "survival",
        "Red flags",
        "config hash",
        "git SHA",
    ):
        assert token in md, token


def test_honest_trial_count_matches_logger(outcome) -> None:  # type: ignore[no-untyped-def]
    md = (outcome.run_dir / "report.md").read_text(encoding="utf-8")
    # The DSR trial count is the SELECTION search size: exactly the 8 swept combos
    # (the confirmation backtests of the winner use a separate logger, by design).
    assert outcome.n_trials == 8
    assert f"**{outcome.n_trials}**" in md


def test_verdict_json_is_consistent(outcome) -> None:  # type: ignore[no-untyped-def]
    data = json.loads((outcome.run_dir / "verdict.json").read_text(encoding="utf-8"))
    assert data["decision"] == "NO-GO"
    names = {g["name"] for g in data["gates"]}
    assert names == {"survival", "deflated_sharpe", "pbo", "no_fail_flags"}


def test_run_is_deterministic(tmp_path: Path) -> None:
    # Two runs with the same seed + fixed created_at produce identical reports.
    def run(sub: str) -> str:
        out = tmp_path / sub
        out.mkdir()
        oc = run_pipeline(
            _DEMO / "config.yaml",
            settings_dir=_CONFIGS,
            snapshot_root=_SNAPSHOTS,
            out_dir=out,
            repo=_REPO,
        )
        return (oc.run_dir / "report.md").read_text(encoding="utf-8")

    assert run("a") == run("b")
