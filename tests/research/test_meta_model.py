"""Tests for the meta-labeling pipeline (G5/G6).

LightGBM primary + L2-logistic baseline are trained on triple-barrier
meta-labels (does the primary signal's directional bet win?), with the T4
uniqueness weights passed straight through as ``sample_weight``. Evaluation runs
*only* through the injected T3 splitter -- there is no plain-k-fold code path --
and every fit writes exactly one TrialRecord. Synthetic fixtures match the T4
Labels/weights schema; nothing is imported from ``futures_engine.labels``.
"""

from __future__ import annotations

from datetime import UTC, datetime

import joblib
import numpy as np
import pandas as pd
import pytest

from futures_engine.research.meta_model import FitResult, MetaModelPipeline
from futures_engine.research.strategies.momentum import DonchianBreakout
from futures_engine.trials.logger import TrialLogger
from futures_engine.validation.splitters import PurgedKFold
from tests.research.conftest import synthetic_mes_1min

FIXED_TS = datetime(2026, 7, 26, 12, 0, tzinfo=UTC)
PARAMS = {
    "lightgbm": {"n_estimators": 30, "num_leaves": 7, "min_child_samples": 5},
    "logistic_l2": {"C": 1.0, "max_iter": 2000},
    "primary": {"window": 20},
}


def _dataset(n_bars: int = 6000, seed: int = 0) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """Synthetic (bars, Labels, uniqueness weights) matching the T4 contract."""
    bars = synthetic_mes_1min(n_bars, seed=seed)
    positions = np.arange(60, n_bars - 40, 8)
    horizon = 20
    close = bars["close"].to_numpy()
    t0 = bars.index[positions]
    t1 = bars.index[positions + horizon]
    ret = close[positions + horizon] / close[positions] - 1.0
    eps = 3e-4
    label = np.where(ret > eps, 1, np.where(ret < -eps, -1, 0))
    touch = np.where(label > 0, "pt", np.where(label < 0, "sl", "time"))
    labels = pd.DataFrame(
        {"t1": t1, "label": label.astype(int), "ret": ret, "touch": touch}, index=t0
    )
    rng = np.random.default_rng(seed + 1)
    weights = pd.Series(rng.uniform(0.3, 1.0, size=len(positions)), index=t0)
    return bars, labels, weights


def _pipeline(model: str = "lightgbm", seed: int = 7) -> MetaModelPipeline:
    return MetaModelPipeline(DonchianBreakout(), model, PARAMS, seed=seed)  # type: ignore[arg-type]


def _fit(pipe: MetaModelPipeline, logger: TrialLogger, seed: int = 0) -> FitResult:
    bars, labels, weights = _dataset(seed=seed)
    return pipe.fit(
        bars,
        labels,
        weights,
        PurgedKFold(n_splits=4, embargo_frac=0.01),
        logger=logger,
        snapshot_hashes=["synthetic-snap"],
        git_sha="testsha0000",
        ts=FIXED_TS,
    )


# --- construction ------------------------------------------------------------


def test_invalid_model_choice_rejected() -> None:
    with pytest.raises(ValueError, match="model"):
        MetaModelPipeline(DonchianBreakout(), "svm", PARAMS, seed=1)  # type: ignore[arg-type]


# --- fit outputs -------------------------------------------------------------


def test_fit_reports_both_models_oos_metrics_side_by_side(tmp_path) -> None:  # type: ignore[no-untyped-def]
    logger = TrialLogger(tmp_path / "t.db")
    result = _fit(_pipeline(), logger)
    assert set(result.oos_metrics) == {"lightgbm", "logistic_l2"}
    for model_metrics in result.oos_metrics.values():
        assert {"accuracy", "auc", "n"} <= set(model_metrics)


def test_per_fold_has_a_row_per_fold_and_model(tmp_path) -> None:  # type: ignore[no-untyped-def]
    logger = TrialLogger(tmp_path / "t.db")
    result = _fit(_pipeline(), logger)
    assert {"fold", "model"} <= set(result.per_fold.columns)
    assert set(result.per_fold["model"].unique()) == {"lightgbm", "logistic_l2"}
    # both models evaluated on every surviving fold.
    counts = result.per_fold.groupby("model")["fold"].nunique()
    assert counts["lightgbm"] == counts["logistic_l2"]
    assert counts["lightgbm"] >= 2


def test_fit_logs_exactly_one_trial_record(tmp_path) -> None:  # type: ignore[no-untyped-def]
    logger = TrialLogger(tmp_path / "t.db")
    _fit(_pipeline(model="lightgbm"), logger)
    assert logger.count() == 1
    rec = logger.all()[0]
    assert rec.strategy_family.startswith("meta")
    assert "lightgbm" in rec.strategy_family
    assert rec.data_snapshot_hashes == ["synthetic-snap"]
    assert rec.seed == 7
    assert rec.git_sha == "testsha0000"
    assert "PurgedKFold" in rec.cv_scheme
    assert len(rec.config_hash) == 64


# --- predict -----------------------------------------------------------------


def test_predict_returns_probabilities_in_unit_interval(tmp_path) -> None:  # type: ignore[no-untyped-def]
    logger = TrialLogger(tmp_path / "t.db")
    pipe = _pipeline()
    _fit(pipe, logger)
    bars, _labels, _w = _dataset()
    proba = pipe.predict(bars)
    assert isinstance(proba, pd.Series)
    assert proba.index.equals(bars.index)
    assert proba.between(0.0, 1.0).all()


def test_predict_before_fit_raises(tmp_path) -> None:  # type: ignore[no-untyped-def]
    bars, _labels, _w = _dataset()
    with pytest.raises(RuntimeError, match="fit"):
        _pipeline().predict(bars)


# --- determinism + weights ---------------------------------------------------


def test_deterministic_under_fixed_seed(tmp_path) -> None:  # type: ignore[no-untyped-def]
    logger_a = TrialLogger(tmp_path / "a.db")
    logger_b = TrialLogger(tmp_path / "b.db")
    pipe_a, pipe_b = _pipeline(seed=13), _pipeline(seed=13)
    res_a, res_b = _fit(pipe_a, logger_a), _fit(pipe_b, logger_b)
    assert res_a.oos_metrics == res_b.oos_metrics
    bars, _labels, _w = _dataset()
    pd.testing.assert_series_equal(pipe_a.predict(bars), pipe_b.predict(bars))


def test_uniqueness_weights_flow_into_training(tmp_path) -> None:  # type: ignore[no-untyped-def]
    # Same data + seed; only the sample weights differ -> predictions must differ,
    # proving the weights are actually used as sample_weight.
    bars, labels, weights = _dataset()
    splitter = PurgedKFold(n_splits=4, embargo_frac=0.01)

    pipe_uniform = _pipeline(seed=5)
    pipe_uniform.fit(
        bars,
        labels,
        pd.Series(1.0, index=weights.index),
        splitter,
        logger=TrialLogger(tmp_path / "u.db"),
        git_sha="x",
        ts=FIXED_TS,
    )
    skewed = pd.Series(np.linspace(0.01, 1.0, len(weights)), index=weights.index)
    pipe_skewed = _pipeline(seed=5)
    pipe_skewed.fit(
        bars,
        labels,
        skewed,
        splitter,
        logger=TrialLogger(tmp_path / "s.db"),
        git_sha="x",
        ts=FIXED_TS,
    )
    assert not np.allclose(
        pipe_uniform.predict(bars).to_numpy(), pipe_skewed.predict(bars).to_numpy()
    )


# --- model artifact + splitter injection -------------------------------------


# joblib.load trips a numpy-2.5 DeprecationWarning inside joblib itself (third
# party, not our code); scope the ignore to this one artifact round-trip test.
@pytest.mark.filterwarnings("ignore:Setting the shape on a NumPy array:DeprecationWarning")
def test_model_artifact_saved_and_loadable(tmp_path) -> None:  # type: ignore[no-untyped-def]
    logger = TrialLogger(tmp_path / "t.db")
    pipe = _pipeline(model="lightgbm")
    result = pipe.fit(
        *_dataset(),
        PurgedKFold(n_splits=4, embargo_frac=0.01),
        logger=logger,
        artifact_dir=tmp_path / "artifacts",
        git_sha="x",
        ts=FIXED_TS,
    )
    assert result.model_artifact.exists()
    loaded = joblib.load(result.model_artifact)
    assert hasattr(loaded, "predict_proba")


def test_evaluation_uses_injected_splitter_folds(tmp_path) -> None:  # type: ignore[no-untyped-def]
    logger = TrialLogger(tmp_path / "t.db")
    bars, labels, weights = _dataset()
    result = _pipeline().fit(
        bars,
        labels,
        weights,
        PurgedKFold(n_splits=6, embargo_frac=0.0),
        logger=logger,
        git_sha="x",
        ts=FIXED_TS,
    )
    # per-fold rows come straight from the injected splitter (<= its fold count).
    assert result.per_fold["fold"].nunique() <= 6
    assert result.per_fold["fold"].nunique() >= 3
