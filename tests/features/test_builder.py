"""Tests for the feature builder: registry, values, intraday annualization, config.

Guards that the assembled matrix's columns are exactly ``FEATURE_COLUMNS`` (the
explicit registered-feature list), that ported feature values reproduce the
legacy definitions, that the volatility annualization factor is derived from the
bar interval (not a hardcoded 252), and that the pydantic feature config is
frozen and rejects unknown keys (no magic constants).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from futures_engine.core.types import InstrumentSpec
from futures_engine.features import indicators as ind
from futures_engine.features.builder import (
    FEATURE_COLUMNS,
    FeatureConfig,
    build_features,
    feature_functions,
    periods_per_year,
)


@pytest.fixture
def spec() -> InstrumentSpec:
    return InstrumentSpec(
        symbol_root="MES",
        exchange="CME",
        tick_size=0.25,
        tick_value=1.25,
        multiplier=5.0,
        currency="USD",
    )


def test_columns_are_exactly_feature_columns(
    bars_fixture: pd.DataFrame, spec: InstrumentSpec
) -> None:
    out = build_features(bars_fixture, spec, FeatureConfig())
    assert list(out.columns) == list(FEATURE_COLUMNS)


def test_feature_columns_match_default_config_keys() -> None:
    """The explicit FEATURE_COLUMNS literal must equal what the default config builds."""
    assert tuple(feature_functions(FeatureConfig()).keys()) == FEATURE_COLUMNS


def test_index_is_preserved(bars_fixture: pd.DataFrame, spec: InstrumentSpec) -> None:
    out = build_features(bars_fixture, spec, FeatureConfig())
    assert out.index.equals(bars_fixture.index)


def test_default_feature_set_has_sixteen_columns() -> None:
    assert len(FEATURE_COLUMNS) == 16  # legacy 15 + fracdiff_close


def test_rsi_feature_matches_indicator_scaled(
    bars_fixture: pd.DataFrame, spec: InstrumentSpec
) -> None:
    out = build_features(bars_fixture, spec, FeatureConfig())
    expected = ind.rsi(bars_fixture["close"], 14) / 100.0
    pd.testing.assert_series_equal(out["rsi"], expected, check_names=False)


def test_return_feature_matches_pct_change(
    bars_fixture: pd.DataFrame, spec: InstrumentSpec
) -> None:
    out = build_features(bars_fixture, spec, FeatureConfig())
    expected = bars_fixture["close"].pct_change(5, fill_method=None)
    pd.testing.assert_series_equal(out["ret_5"], expected, check_names=False)


def test_dist_sma_feature_matches_legacy(bars_fixture: pd.DataFrame, spec: InstrumentSpec) -> None:
    out = build_features(bars_fixture, spec, FeatureConfig())
    expected = bars_fixture["close"] / ind.sma(bars_fixture["close"], 50) - 1.0
    pd.testing.assert_series_equal(out["dist_sma_fast"], expected, check_names=False)


def test_periods_per_year_by_interval() -> None:
    assert periods_per_year("1d", 252.0, 23.0) == pytest.approx(252.0)
    assert periods_per_year("1h", 252.0, 23.0) == pytest.approx(252.0 * 23.0)
    assert periods_per_year("15m", 252.0, 23.0) == pytest.approx(252.0 * 23.0 * 60 / 15)
    assert periods_per_year("5m", 252.0, 23.0) == pytest.approx(252.0 * 23.0 * 60 / 5)
    assert periods_per_year("1m", 252.0, 23.0) == pytest.approx(252.0 * 23.0 * 60)


def test_realized_vol_annualization_follows_interval(
    bars_fixture: pd.DataFrame, spec: InstrumentSpec
) -> None:
    daily = build_features(bars_fixture, spec, FeatureConfig(interval="1d"))
    hourly = build_features(bars_fixture, spec, FeatureConfig(interval="1h"))
    ratio = (hourly["realized_vol"] / daily["realized_vol"]).dropna()
    assert np.allclose(ratio, np.sqrt(23.0))


def test_config_is_frozen_and_forbids_extra() -> None:
    cfg = FeatureConfig()
    with pytest.raises(ValidationError):
        FeatureConfig(unknown_param=1)  # type: ignore[call-arg]
    with pytest.raises(ValidationError):
        cfg.rsi_window = 7  # type: ignore[misc]


def test_config_windows_drive_features(bars_fixture: pd.DataFrame, spec: InstrumentSpec) -> None:
    out = build_features(bars_fixture, spec, FeatureConfig(rsi_window=7))
    expected = ind.rsi(bars_fixture["close"], 7) / 100.0
    pd.testing.assert_series_equal(out["rsi"], expected, check_names=False)


def test_missing_bar_column_raises(spec: InstrumentSpec) -> None:
    bad = pd.DataFrame({"close": [1.0, 2.0, 3.0]})
    with pytest.raises(ValueError, match="MES"):
        build_features(bad, spec, FeatureConfig())
