"""Tests for the typed config loader (task T0 acceptance: config validation)."""

from __future__ import annotations

import math
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from futures_engine.core.config import (
    Settings,
    load_instruments,
    load_yaml,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIGS_DIR = REPO_ROOT / "configs"


def _write_yaml(tmp_path: Path, payload: object) -> Path:
    path = tmp_path / "instruments.yaml"
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    return path


# --- real config file --------------------------------------------------------


def test_real_instruments_file_yields_mes_and_mnq() -> None:
    instruments = load_instruments(CONFIGS_DIR / "instruments.yaml")

    assert set(instruments) == {"MES", "MNQ"}

    mes = instruments["MES"]
    assert mes.exchange == "CME"
    assert mes.tick_size == 0.25
    assert mes.tick_value == 1.25
    assert mes.multiplier == 5
    assert mes.currency == "USD"

    mnq = instruments["MNQ"]
    assert mnq.tick_size == 0.25
    assert mnq.tick_value == 0.50
    assert mnq.multiplier == 2

    # Tick economics invariant holds for every loaded instrument.
    for spec in instruments.values():
        assert math.isclose(spec.tick_value, spec.multiplier * spec.tick_size)


def test_settings_load_reads_instruments() -> None:
    settings = Settings.load(CONFIGS_DIR)
    assert set(settings.instruments) == {"MES", "MNQ"}


# --- validation: unknown keys ------------------------------------------------


def test_unknown_key_inside_instrument_is_rejected(tmp_path: Path) -> None:
    path = _write_yaml(
        tmp_path,
        {
            "instruments": [
                {
                    "symbol_root": "MES",
                    "exchange": "CME",
                    "tick_size": 0.25,
                    "tick_value": 1.25,
                    "multiplier": 5,
                    "currency": "USD",
                    "bogus_field": 1,  # unknown key
                }
            ]
        },
    )
    with pytest.raises(ValidationError):
        load_instruments(path)


def test_unknown_top_level_key_is_rejected(tmp_path: Path) -> None:
    path = _write_yaml(
        tmp_path,
        {
            "instruments": [],
            "unexpected_section": {},  # unknown top-level key
        },
    )
    with pytest.raises(ValidationError):
        load_instruments(path)


# --- validation: invalid values ----------------------------------------------


def test_inconsistent_tick_economics_is_rejected(tmp_path: Path) -> None:
    path = _write_yaml(
        tmp_path,
        {
            "instruments": [
                {
                    "symbol_root": "MES",
                    "exchange": "CME",
                    "tick_size": 0.25,
                    "tick_value": 9.99,  # != multiplier * tick_size
                    "multiplier": 5,
                    "currency": "USD",
                }
            ]
        },
    )
    with pytest.raises(ValidationError):
        load_instruments(path)


def test_non_positive_tick_size_is_rejected(tmp_path: Path) -> None:
    path = _write_yaml(
        tmp_path,
        {
            "instruments": [
                {
                    "symbol_root": "MES",
                    "exchange": "CME",
                    "tick_size": 0.0,  # must be > 0
                    "tick_value": 0.0,
                    "multiplier": 5,
                    "currency": "USD",
                }
            ]
        },
    )
    with pytest.raises(ValidationError):
        load_instruments(path)


def test_duplicate_symbol_root_is_rejected(tmp_path: Path) -> None:
    spec = {
        "symbol_root": "MES",
        "exchange": "CME",
        "tick_size": 0.25,
        "tick_value": 1.25,
        "multiplier": 5,
        "currency": "USD",
    }
    path = _write_yaml(tmp_path, {"instruments": [spec, dict(spec)]})
    with pytest.raises(ValueError, match="duplicate"):
        load_instruments(path)


def test_non_mapping_top_level_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "instruments.yaml"
    path.write_text(yaml.safe_dump(["not", "a", "mapping"]), encoding="utf-8")
    with pytest.raises(ValueError, match="mapping"):
        load_yaml(path)
