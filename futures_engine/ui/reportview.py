"""Run submission + report reading for the research cockpit (U5).

This is the integration seam between the Configure handoff and the report view.
Two responsibilities, both thin (UI-G1):

* :func:`submit_run` loads the handed-off :class:`PipelineConfig` and enqueues the
  REAL :func:`~futures_engine.pipeline.run.run_pipeline` on the U2
  :class:`~futures_engine.ui.jobs.JobRegistry` process pool (UI-G5). Because
  ``ProcessPoolExecutor`` pickles the submitted callable, the work is a
  module-level function (:func:`_run_pipeline_job`, all ``str`` args) bound via
  :func:`functools.partial` -- never a closure or lambda. Its return value is a
  small picklable summary dict; the report is read from disk regardless.
* :func:`load_report` parses a finished run's ON-DISK artifacts
  (``verdict.json`` + ``manifest.json`` + ``report.html``) into a
  :class:`ReportContext`. It recomputes NOTHING. Missing artifacts (a failed or
  never-run job) yield an ``incomplete`` context, never a crash.
"""

from __future__ import annotations

import functools
import json
from dataclasses import dataclass, field
from pathlib import Path

from futures_engine.pipeline.run import PipelineConfig, run_pipeline
from futures_engine.report.builder import GateResult, Verdict
from futures_engine.ui.jobs import JobRegistry, get_registry

# Artifacts a complete run writes into its run dir (see report.builder.build_report).
_ARTIFACTS: tuple[str, ...] = ("verdict.json", "report.html", "manifest.json")


def _run_pipeline_job(
    config_path: str,
    settings_dir: str,
    snapshot_root: str,
    out_dir: str,
) -> dict[str, str]:
    """Top-level, picklable pool job: run the pipeline and return a small summary.

    All parameters are plain ``str`` paths so the bound callable pickles cleanly
    for ``ProcessPoolExecutor`` (UI-G5). The returned dict is picklable; the report
    view reads the run's artifacts from disk, so this summary is advisory only.
    """
    outcome = run_pipeline(
        config_path,
        settings_dir=settings_dir,
        snapshot_root=snapshot_root,
        out_dir=out_dir,
    )
    return {
        "run_id": outcome.run_id,
        "run_dir": str(outcome.run_dir),
        "decision": outcome.decision,
    }


def submit_run(
    cfg_path: str | Path,
    *,
    settings_dir: str | Path,
    snapshot_root: str | Path,
    out_dir: str | Path,
    registry: JobRegistry | None = None,
) -> str:
    """Enqueue ``run_pipeline`` for the config at ``cfg_path``; return the job id.

    The job id IS the config's ``run_id`` (the pipeline writes ``out_dir/<run_id>/``),
    so the run is addressable by a single identifier across the routes. The submitted
    callable is a :func:`functools.partial` of the module-level :func:`_run_pipeline_job`
    with ``str`` paths -- picklable for the process pool (UI-G5).
    """
    run_id = PipelineConfig.load(cfg_path).run_id
    reg = registry if registry is not None else get_registry()
    job = functools.partial(
        _run_pipeline_job,
        str(cfg_path),
        str(settings_dir),
        str(snapshot_root),
        str(out_dir),
    )
    return reg.submit(job, job_id=run_id)


@dataclass(frozen=True, slots=True)
class ReportContext:
    """View model for a run's report, parsed from on-disk artifacts (no recompute).

    ``complete`` is ``True`` only when ``verdict.json`` was read successfully. An
    incomplete context (failed / never-run job) still renders a graceful page.
    """

    run_id: str
    complete: bool
    decision: str | None = None
    gates: list[GateResult] = field(default_factory=list)
    report_html: str | None = None
    config_hash: str | None = None
    snapshot_hashes: list[str] = field(default_factory=list)
    git_sha: str | None = None
    seed: int | None = None
    created_at: str | None = None
    missing: list[str] = field(default_factory=list)


def load_report(run_dir: str | Path) -> ReportContext:
    """Parse a run's artifacts into a :class:`ReportContext` (UI-G1: no recompute).

    Reads ``verdict.json`` (decision + gate rows), ``manifest.json`` (provenance
    hashes) and embeds our own ``report.html``. Any missing artifact is recorded in
    ``missing`` and yields an ``incomplete`` context rather than raising.
    """
    path = Path(run_dir)
    run_id = path.name
    missing = [name for name in _ARTIFACTS if not (path / name).is_file()]

    verdict_path = path / "verdict.json"
    if not verdict_path.is_file():
        return ReportContext(run_id=run_id, complete=False, missing=missing)

    verdict = Verdict.model_validate_json(verdict_path.read_text(encoding="utf-8"))

    manifest_path = path / "manifest.json"
    manifest: dict[str, object] = {}
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    report_html_path = path / "report.html"
    report_html = (
        report_html_path.read_text(encoding="utf-8") if report_html_path.is_file() else None
    )

    snapshots = manifest.get("data_snapshot_hashes", [])
    seed = manifest.get("seed")
    created_at = manifest.get("created_at")
    return ReportContext(
        run_id=run_id,
        complete=True,
        decision=verdict.decision,
        gates=list(verdict.gates),
        report_html=report_html,
        config_hash=_as_str(manifest.get("config_hash")),
        snapshot_hashes=[str(h) for h in snapshots] if isinstance(snapshots, list) else [],
        git_sha=_as_str(manifest.get("git_sha")),
        seed=seed if isinstance(seed, int) else None,
        created_at=_as_str(created_at),
        missing=missing,
    )


def _as_str(value: object) -> str | None:
    return str(value) if value is not None else None
