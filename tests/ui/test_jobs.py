"""Tests for the process-pool job registry (U2).

Deterministic and offline: jobs are trivial module-level (picklable) callables,
and we poll each job to a terminal state with a bounded loop -- no wall-clock
assertions beyond ``list()`` newest-first ordering. The registry owns a
``ProcessPoolExecutor`` (UI-G5), so each test builds a *fresh* registry and
shuts it down to avoid leaking processes.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from functools import partial

import pytest

from futures_engine.ui.jobs import JobRecord, JobRegistry, get_registry

# --- Module-level (picklable) job functions ---------------------------------


def _return_42() -> int:
    return 42


def _raise_boom() -> int:
    raise ValueError("boom-42")


def _sleep_then_return(seconds: float) -> str:
    time.sleep(seconds)
    return "slept"


@pytest.fixture
def registry() -> Iterator[JobRegistry]:
    reg = JobRegistry(max_workers=2)
    try:
        yield reg
    finally:
        reg.shutdown()


def _poll_terminal(reg: JobRegistry, job_id: str, *, timeout: float = 30.0) -> JobRecord:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        rec = reg.get(job_id)
        if rec.status in ("done", "failed"):
            return rec
        time.sleep(0.01)
    raise AssertionError(f"job {job_id} did not reach a terminal state in {timeout}s")


# --- Done path --------------------------------------------------------------


def test_submit_trivial_job_transitions_to_done_with_result(
    registry: JobRegistry,
) -> None:
    job_id = registry.submit(_return_42)
    rec = _poll_terminal(registry, job_id)
    assert rec.status == "done"
    assert rec.result == 42
    assert rec.error is None
    assert rec.started_at is not None
    assert rec.finished_at is not None


def test_queued_status_is_representable() -> None:
    # A record starts life 'queued' before the pool picks it up.
    reg = JobRegistry(max_workers=1)
    try:
        job_id = reg.submit(_return_42)
        # Status is one of the four literals immediately after submit.
        assert reg.get(job_id).status in ("queued", "running", "done")
    finally:
        reg.shutdown()


# --- Running observability --------------------------------------------------


def test_running_state_is_observable(registry: JobRegistry) -> None:
    """A slow job is observably 'running' before it completes (U5 progress UI)."""
    job_id = registry.submit(partial(_sleep_then_return, 0.5))
    saw_running = False
    deadline = time.monotonic() + 30.0
    while time.monotonic() < deadline:
        status = registry.get(job_id).status
        if status == "running":
            saw_running = True
            break
        if status in ("done", "failed"):
            break
        time.sleep(0.005)
    assert saw_running, "job never entered an observable 'running' state"
    rec = _poll_terminal(registry, job_id)
    assert rec.status == "done"
    assert rec.result == "slept"


# --- Failed (raising fn) path -----------------------------------------------


def test_raising_job_transitions_to_failed_with_traceback(
    registry: JobRegistry,
) -> None:
    job_id = registry.submit(_raise_boom)
    rec = _poll_terminal(registry, job_id)
    assert rec.status == "failed"
    assert rec.result is None
    assert rec.error is not None
    assert "ValueError" in rec.error
    assert "boom-42" in rec.error


# --- Non-picklable fn -> failed, not a hang ---------------------------------


def test_non_picklable_job_fails_cleanly_without_hang(registry: JobRegistry) -> None:
    # A lambda cannot be pickled to a worker process; this must surface as a
    # 'failed' job with a clear error -- never a hang or a crashed pool.
    job_id = registry.submit(lambda: 1)  # type: ignore[arg-type,return-value]
    rec = _poll_terminal(registry, job_id, timeout=30.0)
    assert rec.status == "failed"
    assert rec.error is not None
    assert rec.error != ""
    # The pool must still be usable after a picklability failure.
    ok_id = registry.submit(_return_42)
    assert _poll_terminal(registry, ok_id).result == 42


# --- Registry bookkeeping ---------------------------------------------------


def test_get_unknown_id_raises_keyerror(registry: JobRegistry) -> None:
    with pytest.raises(KeyError):
        registry.get("does-not-exist")


def test_explicit_job_id_is_used(registry: JobRegistry) -> None:
    job_id = registry.submit(_return_42, job_id="my-fixed-id")
    assert job_id == "my-fixed-id"
    assert registry.get("my-fixed-id").id == "my-fixed-id"


def test_list_returns_records_newest_first(registry: JobRegistry) -> None:
    first = registry.submit(_return_42, job_id="j1")
    second = registry.submit(_return_42, job_id="j2")
    third = registry.submit(_return_42, job_id="j3")
    ids = [rec.id for rec in registry.list()]
    assert ids[:3] == [third, second, first]


# --- Singleton accessor -----------------------------------------------------


def test_get_registry_returns_singleton() -> None:
    assert get_registry() is get_registry()
