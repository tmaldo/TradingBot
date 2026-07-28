"""``configs/live.yaml`` loads and validates through the pydantic pattern (G15).

Unknown keys are rejected; the sizing/edge inputs round-trip into the T7 value
objects the RiskManager consumes.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from futures_engine.execution.live_config import LiveConfig
from futures_engine.sizing.position import EdgeStats, SizingConfig, position_size

from .conftest import MES

CONFIGS_DIR = Path(__file__).resolve().parents[2] / "configs"


def test_live_yaml_loads_and_validates() -> None:
    cfg = LiveConfig.load(CONFIGS_DIR / "live.yaml")
    assert cfg.account.prop_preset == "topstep_50k"
    assert cfg.risk.kill_switches.max_order_rate.max_per_minute == 10
    assert cfg.shutdown.rolling_sharpe_floor == 0.25
    assert cfg.adapters.tradovate.account_id == 1000001


def test_unknown_key_is_rejected() -> None:
    cfg = LiveConfig.load(CONFIGS_DIR / "live.yaml")
    raw = cfg.model_dump()
    raw["account"]["surprise"] = 1  # a typo/extra key
    with pytest.raises(ValidationError):
        LiveConfig.model_validate(raw)


def test_sizing_inputs_round_trip_into_t7_objects() -> None:
    cfg = LiveConfig.load(CONFIGS_DIR / "live.yaml")
    s = cfg.risk.sizing
    assert isinstance(s.edge.to_edge_stats(), EdgeStats)
    assert isinstance(s.to_sizing_config(), SizingConfig)
    cap = position_size(
        s.vol_estimate_usd,
        s.edge.to_edge_stats(),
        MES,
        s.to_sizing_config(),
        s.survival_max_contracts,
    )
    # the demo config's binding (quarter-Kelly) leg yields a 3-contract cap.
    assert cap == 3


def test_kelly_cap_above_quarter_rejected_in_live_config() -> None:
    cfg = LiveConfig.load(CONFIGS_DIR / "live.yaml")
    raw = cfg.model_dump()
    raw["risk"]["sizing"]["kelly_fraction_cap"] = 0.5  # above quarter-Kelly (G12)
    with pytest.raises(ValidationError):
        LiveConfig.model_validate(raw)
