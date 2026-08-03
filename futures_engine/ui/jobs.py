"""In-process job registry that runs work in a ``ProcessPoolExecutor``.

This is the async backbone of the research cockpit. The cockpit's pipeline
sweep is CPU-bound (pandas holds the GIL), so it must never run on the FastAPI
event loop or a thread pool (UI-G5). :class:`JobRegistry` submits generic
callables to a **process** pool and tracks their status in an in-memory,
lock-guarded registry so the web layer can poll progress without blocking.

This module is deliberately pure stdlib -- no ``fastapi`` import (UI-G1/UI-G2).
It is a generic runner: the pipeline wiring lives in a later task (U5), which
passes a top-level, picklable callable (e.g. a :func:`functools.partial` of a
module-level run function).

Running-state observability
---------------------------
``concurrent.futures`` offers no "execution started" callback, so we cannot
stamp the exact instant a worker picks up a job. Instead we *reconcile* status
on read: a non-terminal record is reported ``running`` when its future reports
:meth:`~concurrent.futures.Future.running`, otherwise ``queued``. Terminal
states (``done``/``failed``) are set authoritatively by a
:meth:`~concurrent.futures.Future.add_done_callback` callback. This makes a
``running`` state reliably representable and observable for U5's progress UI
without a per-job monitor thread.
"""

from __future__ import annotations

import threading
import traceback
from collections.abc import Callable
from concurrent.futures import Future, ProcessPoolExecutor
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from typing import Literal, TypeVar
from uuid import uuid4

T = TypeVar("T")

JobStatus = Literal["queued", "running", "done", "failed"]

_TERMINAL: frozenset[JobStatus] = frozenset({"done", "failed"})


def _utcnow() -> datetime:
    return datetime.now(UTC)


@dataclass(frozen=True, slots=True)
class JobRecord:
    """Immutable snapshot of a job's state.

    ``result`` is populated only on ``done``; ``error`` only on ``failed`` and
    holds a formatted traceback string. ``started_at``/``finished_at`` are
    best-effort UTC stamps (``started_at`` is set when the job is first observed
    running, or at completion if that transition was never seen).
    """

    id: str
    status: JobStatus
    result: object | None = None
    error: str | None = None
    started_at: datetime | None = None
    finished_at: datetime | None = None


class JobRegistry:
    """Runs submitted callables in a process pool and tracks their status.

    The pool is created lazily on first :meth:`submit` and owned for the life of
    the registry. Records live in an in-memory dict guarded by a lock. Instances
    are independent, so tests can build a throwaway registry and
    :meth:`shutdown` it to avoid leaking worker processes.
    """

    def __init__(self, *, max_workers: int | None = None) -> None:
        self._max_workers = max_workers
        self._lock = threading.Lock()
        self._records: dict[str, JobRecord] = {}
        self._futures: dict[str, Future[object]] = {}
        self._order: list[str] = []
        self._pool: ProcessPoolExecutor | None = None

    # -- pool lifecycle ------------------------------------------------------

    def _ensure_pool(self) -> ProcessPoolExecutor:
        # Caller holds the lock.
        if self._pool is None:
            self._pool = ProcessPoolExecutor(max_workers=self._max_workers)
        return self._pool

    def shutdown(self, *, wait: bool = True) -> None:
        """Shut the process pool down (and, by default, wait for workers)."""
        with self._lock:
            pool, self._pool = self._pool, None
        if pool is not None:
            pool.shutdown(wait=wait)

    # -- submission ----------------------------------------------------------

    def submit(self, fn: Callable[[], T], *, job_id: str | None = None) -> str:
        """Submit ``fn`` to the process pool and return its job id.

        ``fn`` must be top-level/picklable (a lambda or closure will fail to
        pickle). A picklability failure surfaces as a ``failed`` job with a
        clear error -- never a hang.
        """
        jid = job_id if job_id is not None else uuid4().hex
        record = JobRecord(id=jid, status="queued")
        with self._lock:
            if jid in self._records:
                raise ValueError(f"duplicate job id: {jid!r}")
            self._records[jid] = record
            self._order.append(jid)
            pool = self._ensure_pool()
            future: Future[object] = pool.submit(fn)
            self._futures[jid] = future
        future.add_done_callback(lambda fut: self._on_done(jid, fut))
        return jid

    def _on_done(self, jid: str, future: Future[object]) -> None:
        finished = _utcnow()
        error = future.exception()
        with self._lock:
            record = self._records[jid]
            started = record.started_at or finished
            if error is not None:
                text = "".join(traceback.format_exception(type(error), error, error.__traceback__))
                self._records[jid] = replace(
                    record,
                    status="failed",
                    error=text,
                    started_at=started,
                    finished_at=finished,
                )
            else:
                self._records[jid] = replace(
                    record,
                    status="done",
                    result=future.result(),
                    started_at=started,
                    finished_at=finished,
                )

    # -- reads ---------------------------------------------------------------

    def _reconcile(self, jid: str) -> JobRecord:
        # Caller holds the lock. Derive running/queued for non-terminal jobs
        # from the live future, stamping started_at the first time we see it run.
        record = self._records[jid]
        if record.status in _TERMINAL:
            return record
        future = self._futures.get(jid)
        if future is not None and future.running() and record.status != "running":
            record = replace(
                record,
                status="running",
                started_at=record.started_at or _utcnow(),
            )
            self._records[jid] = record
        return record

    def get(self, job_id: str) -> JobRecord:
        """Return the current record for ``job_id`` (raises ``KeyError``)."""
        with self._lock:
            if job_id not in self._records:
                raise KeyError(job_id)
            return self._reconcile(job_id)

    def list(self) -> list[JobRecord]:
        """Return all records, newest submission first."""
        with self._lock:
            return [self._reconcile(jid) for jid in reversed(self._order)]


_registry: JobRegistry | None = None
_registry_lock = threading.Lock()


def get_registry() -> JobRegistry:
    """Return the process-wide :class:`JobRegistry` singleton."""
    global _registry
    with _registry_lock:
        if _registry is None:
            _registry = JobRegistry()
        return _registry
