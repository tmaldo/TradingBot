"""Tests for RunManifest (task T0 acceptance: JSON round-trip + git SHA)."""

from __future__ import annotations

import re
from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic import ValidationError

from futures_engine.core.manifest import RunManifest, current_git_sha

REPO_ROOT = Path(__file__).resolve().parents[1]
CREATED_AT = datetime(2026, 7, 25, 12, 0, 0, tzinfo=UTC)
_SHA_RE = re.compile(r"^[0-9a-f]{40}$")


def _manifest(**overrides: object) -> RunManifest:
    kwargs: dict[str, object] = {
        "run_id": "run-1",
        "created_at": CREATED_AT,
        "git_sha": "0" * 40,
        "config_hash": "cfg-1",
        "data_snapshot_hashes": ["snap-1"],
        "seed": 42,
        "trial_ids": ["t-1", "t-2"],
    }
    kwargs.update(overrides)
    return RunManifest(**kwargs)


def test_round_trips_to_json() -> None:
    manifest = _manifest()
    restored = RunManifest.model_validate_json(manifest.model_dump_json())
    assert restored == manifest
    assert restored.created_at == CREATED_AT


def test_unknown_field_rejected() -> None:
    with pytest.raises(ValidationError):
        RunManifest.model_validate(
            {
                "run_id": "r",
                "created_at": CREATED_AT,
                "git_sha": "0" * 40,
                "config_hash": "c",
                "data_snapshot_hashes": [],
                "seed": 1,
                "trial_ids": [],
                "surprise": True,
            }
        )


def test_current_git_sha_reads_repo_head() -> None:
    sha = current_git_sha(REPO_ROOT)
    assert _SHA_RE.match(sha), f"not a full git sha: {sha!r}"


def test_create_pulls_git_sha_from_repo() -> None:
    manifest = RunManifest.create(
        created_at=CREATED_AT,
        config_hash="cfg-1",
        data_snapshot_hashes=["snap-1"],
        seed=42,
        repo=REPO_ROOT,
    )
    assert _SHA_RE.match(manifest.git_sha)
    assert manifest.git_sha == current_git_sha(REPO_ROOT)
    # A run_id is generated when not supplied.
    assert manifest.run_id
    assert manifest.trial_ids == []


def test_create_generates_unique_run_ids() -> None:
    a = RunManifest.create(
        created_at=CREATED_AT,
        config_hash="c",
        data_snapshot_hashes=[],
        seed=1,
        git_sha="0" * 40,
    )
    b = RunManifest.create(
        created_at=CREATED_AT,
        config_hash="c",
        data_snapshot_hashes=[],
        seed=1,
        git_sha="0" * 40,
    )
    assert a.run_id != b.run_id
