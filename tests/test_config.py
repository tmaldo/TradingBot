"""Tests for the typed config loader (task T0 acceptance: config validation)."""

from __future__ import annotations

import math
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from futures_engine.core.config import (
    Settings,
    load_costs,
    load_instruments,
    load_prop_rules,
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


# --- costs + prop rules (task T2) --------------------------------------------


def test_settings_load_reads_costs_and_prop_rules() -> None:
    settings = Settings.load(CONFIGS_DIR)
    assert set(settings.costs) == {"MES", "MNQ"}
    assert set(settings.prop_rules) == {"topstep_50k", "mffu_50k", "apex_50k"}


def test_real_cost_profiles_all_in_round_turn_in_band() -> None:
    costs = load_costs(CONFIGS_DIR / "costs.yaml")
    for cfg in costs.values():
        all_in_rt = (
            cfg.commission_per_side_usd + cfg.exchange_fee_per_side_usd + cfg.nfa_fee_per_side_usd
        ) * 2
        assert 1.02 <= all_in_rt <= 1.04


def test_real_prop_presets_match_published_rules() -> None:
    presets = load_prop_rules(CONFIGS_DIR / "prop_rules.yaml")

    topstep = presets["topstep_50k"]
    assert topstep.name == "topstep_50k"  # name injected from the mapping key
    assert topstep.trailing_mode == "eod"
    assert topstep.trailing_freezes_at_start_balance is True
    assert topstep.daily_loss_limit == 1000.0
    assert topstep.profit_target == 3000.0

    # Apex evaluation uses intraday-unrealized trailing and does not freeze.
    apex = presets["apex_50k"]
    assert apex.trailing_mode == "intraday_unrealized"
    assert apex.trailing_freezes_at_start_balance is False
    assert apex.daily_loss_limit is None

    # MFFU has no daily loss limit but a 50% consistency rule and a 2-day minimum.
    mffu = presets["mffu_50k"]
    assert mffu.daily_loss_limit is None
    assert mffu.consistency_max_day_pct == 0.50
    assert mffu.min_trading_days == 2


def test_unknown_key_in_cost_profile_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "costs.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "costs": {
                    "MES": {
                        "commission_per_side_usd": 0.15,
                        "exchange_fee_per_side_usd": 0.35,
                        "nfa_fee_per_side_usd": 0.01,
                        "spread_ticks": 1.0,
                        "slippage": "fixed_ticks",
                        "slippage_ticks": 1.0,
                        "delay_bars": 0,
                        "bogus": 1,  # unknown key
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValidationError):
        load_costs(path)


def test_unknown_key_in_prop_rule_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "prop_rules.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "prop_rules": {
                    "x_50k": {
                        "start_balance": 50000.0,
                        "trailing_dd": 2000.0,
                        "trailing_mode": "eod",
                        "trailing_freezes_at_start_balance": True,
                        "profit_target": 3000.0,
                        "min_trading_days": 0,
                        "bogus": 1,  # unknown key
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValidationError):
        load_prop_rules(path)
