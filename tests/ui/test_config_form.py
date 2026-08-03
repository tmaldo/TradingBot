"""Tests for the cockpit Configure screen: form -> validated PipelineConfig (U4).

Covers the grid parser (valid + malformed), the form->config round-trip with
exact field values, the honest-DSR strategy_family derivation (and that a user
cannot mismatch it), config_to_yaml round-trip, and >=3 invalid cases that
re-render the form inline at HTTP 200 (never a 500).
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from futures_engine.pipeline.run import PipelineConfig
from futures_engine.ui.app import create_app
from futures_engine.ui.config_form import (
    config_to_yaml,
    derive_strategy_family,
    form_to_config,
    parse_grid,
    prop_preset_choices,
    signal_choices,
)

_CONFIGS = Path("configs")


def _valid_form() -> dict[str, str]:
    return {
        "run_id": "cockpit_run",
        "instrument": "MES",
        "signal": "donchian_breakout",
        "grid": "window: 5, 10, 15, 20",
        "snapshot_hash": "abc123def456",
        "prop_preset": "topstep_50k",
        "seed": "7",
        "n_splits": "5",
        "embargo_frac": "0.02",
        "pbo_partitions": "8",
        "bootstrap_n": "500",
        "survival_n_paths": "300",
        "survival_horizon_days": "30",
        "survival_contracts": "1",
        "gate_min_dsr_p": "0.95",
        "gate_max_pbo": "0.5",
        "gate_min_survival": "0.90",
    }


# --- grid parser ------------------------------------------------------------


def test_parse_grid_valid_single_axis() -> None:
    assert parse_grid("window: 5, 10, 15, 20") == {"window": [5, 10, 15, 20]}


def test_parse_grid_valid_multi_axis() -> None:
    grid = parse_grid("fast: 5, 10\nslow: 20, 40")
    assert grid == {"fast": [5, 10], "slow": [20, 40]}


@pytest.mark.parametrize(
    "text",
    [
        "",
        "   \n  ",
        "window 5, 10",  # missing colon
        "window: 5, x, 10",  # non-integer
        "window:",  # no values
        ": 5, 10",  # empty parameter name
    ],
)
def test_parse_grid_malformed_raises(text: str) -> None:
    with pytest.raises(ValueError):
        parse_grid(text)


# --- form -> config round-trip ----------------------------------------------


def test_form_to_config_round_trip_exact_values() -> None:
    cfg = form_to_config(_valid_form())
    assert isinstance(cfg, PipelineConfig)
    assert cfg.run_id == "cockpit_run"
    assert cfg.instrument == "MES"
    assert cfg.signal == "donchian_breakout"
    assert cfg.grid == {"window": [5, 10, 15, 20]}
    assert cfg.snapshot_hash == "abc123def456"
    assert cfg.prop_preset == "topstep_50k"
    assert cfg.seed == 7
    assert cfg.survival.n_paths == 300
    assert cfg.survival.horizon_days == 30
    assert cfg.survival.contracts == 1


def test_form_to_config_gates_default_when_omitted() -> None:
    form = _valid_form()
    del form["gate_min_dsr_p"]
    del form["gate_max_pbo"]
    del form["gate_min_survival"]
    cfg = form_to_config(form)
    assert cfg.gates.min_dsr_p == 0.95
    assert cfg.gates.max_pbo == 0.5
    assert cfg.gates.min_survival == 0.90


def test_form_to_config_gates_override() -> None:
    form = _valid_form()
    form["gate_min_dsr_p"] = "0.99"
    cfg = form_to_config(form)
    assert cfg.gates.min_dsr_p == 0.99


# --- honest-DSR: strategy_family derivation ---------------------------------


def test_strategy_family_derived_from_signal() -> None:
    assert derive_strategy_family("donchian_breakout") == "trend_momentum"
    assert derive_strategy_family("ma_cross_vol_target") == "trend_momentum"


def test_form_to_config_derives_family_ignoring_user_input() -> None:
    form = _valid_form()
    # A user trying to smuggle in a conflicting family MUST NOT win: the value is
    # sourced from the signal registry, never trusted from the form.
    form["strategy_family"] = "mean_reversion_conflict"
    cfg = form_to_config(form)
    assert cfg.strategy_family == "trend_momentum"


def test_form_to_config_unknown_signal_raises() -> None:
    form = _valid_form()
    form["signal"] = "not_a_real_signal"
    with pytest.raises(ValueError):
        form_to_config(form)


# --- config_to_yaml round-trip ----------------------------------------------


def test_config_to_yaml_round_trip(tmp_path: Path) -> None:
    cfg = form_to_config(_valid_form())
    path = tmp_path / "config.yaml"
    config_to_yaml(cfg, path)
    assert path.exists()
    reloaded = PipelineConfig.load(path)
    assert reloaded == cfg


# --- choice helpers ---------------------------------------------------------


def test_signal_choices_matches_registry() -> None:
    choices = signal_choices()
    assert "donchian_breakout" in choices
    assert "ma_cross_vol_target" in choices


def test_prop_preset_choices_from_yaml() -> None:
    presets = prop_preset_choices(_CONFIGS)
    assert set(presets) >= {"topstep_50k", "mffu_50k", "apex_50k"}


# --- routes -----------------------------------------------------------------


@pytest.fixture
def client(tmp_path: Path) -> TestClient:
    return TestClient(create_app(snapshot_root=tmp_path / "snap", runs_root=tmp_path / "runs"))


def test_get_configure_prefills_choices(client: TestClient) -> None:
    body = client.get("/configure").text
    assert "donchian_breakout" in body
    assert "topstep_50k" in body
    assert "MES" in body
    assert "MNQ" in body
    # Honest-DSR: no free-text strategy_family input the user can mismatch.
    assert 'name="strategy_family"' not in body


def test_post_configure_success_persists_and_redirects(client: TestClient, tmp_path: Path) -> None:
    resp = client.post("/configure", data=_valid_form(), follow_redirects=False)
    assert resp.status_code == 303
    assert resp.headers["location"] == "/runs/cockpit_run"
    written = tmp_path / "runs" / "cockpit_run" / "config.yaml"
    assert written.exists()
    cfg = PipelineConfig.load(written)
    assert cfg.run_id == "cockpit_run"
    assert cfg.strategy_family == "trend_momentum"


def test_post_configure_unknown_signal_renders_inline_no_500(client: TestClient) -> None:
    form = _valid_form()
    form["signal"] = "not_a_real_signal"
    resp = client.post("/configure", data=form)
    assert resp.status_code == 200
    assert "not_a_real_signal" in resp.text
    assert "error" in resp.text.lower()


def test_post_configure_malformed_grid_renders_inline_no_500(client: TestClient) -> None:
    form = _valid_form()
    form["grid"] = "window: 5, notanint, 10"
    resp = client.post("/configure", data=form)
    assert resp.status_code == 200
    assert "error" in resp.text.lower()


def test_post_configure_empty_grid_renders_inline_no_500(client: TestClient) -> None:
    form = _valid_form()
    form["grid"] = "   "
    resp = client.post("/configure", data=form)
    assert resp.status_code == 200
    assert "error" in resp.text.lower()


def test_post_configure_out_of_range_gate_renders_inline_no_500(client: TestClient) -> None:
    form = _valid_form()
    form["gate_min_dsr_p"] = "1.5"  # violates le=1.0 on GateConfig
    resp = client.post("/configure", data=form)
    assert resp.status_code == 200
    assert "error" in resp.text.lower()
