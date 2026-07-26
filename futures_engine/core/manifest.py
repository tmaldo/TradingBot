"""Run provenance: the ``RunManifest`` ties a run to its exact inputs.

Reproducibility (Global Constraint G15) requires every run to record its git
SHA, config hash, data snapshot hashes, seed, and the trials it produced.
``created_at`` is passed in explicitly rather than read from the wall clock so
runs remain deterministic and testable (brief: no ``now()`` in library paths).
"""

from __future__ import annotations

import subprocess
import uuid
from datetime import datetime
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field


def current_git_sha(repo: str | Path | None = None) -> str:
    """Return the full HEAD commit SHA of ``repo`` (defaults to cwd).

    Raises ``RuntimeError`` if git is unavailable or the path is not a repo,
    so a missing SHA fails loudly rather than silently producing junk provenance.
    """
    cwd = str(repo) if repo is not None else None
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=cwd,
            capture_output=True,
            text=True,
            check=True,
        )
    except FileNotFoundError as exc:  # git not installed / not on PATH
        raise RuntimeError("git executable not found on PATH") from exc
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            f"could not read git HEAD in {cwd or 'cwd'}: {exc.stderr.strip()}"
        ) from exc
    return completed.stdout.strip()


class RunManifest(BaseModel):
    """Immutable record of everything needed to reproduce a single run."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    run_id: str = Field(min_length=1)
    created_at: datetime
    git_sha: str = Field(min_length=1)
    config_hash: str = Field(min_length=1)
    data_snapshot_hashes: list[str]
    seed: int
    trial_ids: list[str]

    @classmethod
    def create(
        cls,
        *,
        created_at: datetime,
        config_hash: str,
        data_snapshot_hashes: list[str],
        seed: int,
        trial_ids: list[str] | None = None,
        run_id: str | None = None,
        git_sha: str | None = None,
        repo: str | Path | None = None,
    ) -> RunManifest:
        """Build a manifest, defaulting ``git_sha`` from the repo and a fresh ``run_id``.

        ``created_at`` is mandatory and supplied by the caller (reproducibility).
        """
        return cls(
            run_id=run_id if run_id is not None else uuid.uuid4().hex,
            created_at=created_at,
            git_sha=git_sha if git_sha is not None else current_git_sha(repo),
            config_hash=config_hash,
            data_snapshot_hashes=data_snapshot_hashes,
            seed=seed,
            trial_ids=trial_ids if trial_ids is not None else [],
        )
