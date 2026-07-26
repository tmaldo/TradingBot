"""Tests for leak-proof CV splitters (task T3 / G9 / G16 — the hardest tests).

These are the strictest tests in the engine. The central invariant, asserted
exhaustively rather than by spot check, is that **no retained training sample's
label interval overlaps any test sample's label interval** (López de Prado,
AFML §7 purging). We also assert embargo behaviour, the impossibility of
constructing an unpurged configuration, the exact combinatorial structure of
CPCV, and a leakage regression that reproduces the legacy boundary bug
(AUDIT §2.5) and proves purging removes it.
"""

from __future__ import annotations

import itertools
import math

import numpy as np
import pandas as pd
import pytest

from futures_engine.validation.splitters import (
    CombinatorialPurgedCV,
    PurgedKFold,
    WalkForward,
)

# --- helpers -----------------------------------------------------------------


def _overlaps(a0: float, a1: float, b0: float, b1: float) -> bool:
    """Closed-interval overlap test: [a0,a1] meets [b0,b1]."""
    return a0 <= b1 and b0 <= a1


def _fixed_horizon_intervals(n: int, horizon: int) -> pd.Series:
    """Overlapping fixed-horizon labels: sample i spans [i, i+horizon]."""
    t0 = np.arange(n)
    t1 = t0 + horizon
    return pd.Series(t1, index=t0)


def _assert_no_train_test_overlap(
    train_idx: np.ndarray, test_idx: np.ndarray, intervals: pd.Series
) -> None:
    """Exhaustively assert every train interval is disjoint from every test one."""
    t0 = intervals.index.to_numpy()
    t1 = intervals.to_numpy()
    for i in train_idx:
        for j in test_idx:
            assert not _overlaps(t0[i], t1[i], t0[j], t1[j]), (
                f"train sample {i} [{t0[i]},{t1[i]}] overlaps test sample {j} [{t0[j]},{t1[j]}]"
            )


# --- PurgedKFold: structure --------------------------------------------------


def test_purged_kfold_test_folds_partition_all_positions() -> None:
    n, k = 100, 5
    intervals = _fixed_horizon_intervals(n, horizon=3)
    x = np.zeros(n)
    splits = list(PurgedKFold(n_splits=k, embargo_frac=0.0).split(x, intervals))
    assert len(splits) == k
    seen = np.concatenate([test for _, test in splits])
    assert sorted(seen.tolist()) == list(range(n))  # each position tested once
    for _, test in splits:
        # test folds are contiguous position blocks
        assert list(test) == list(range(int(test[0]), int(test[-1]) + 1))


def test_purged_kfold_is_deterministic() -> None:
    n = 60
    intervals = _fixed_horizon_intervals(n, horizon=4)
    x = np.zeros(n)
    cv = PurgedKFold(n_splits=4, embargo_frac=0.1)
    a = [(tr.tolist(), te.tolist()) for tr, te in cv.split(x, intervals)]
    b = [(tr.tolist(), te.tolist()) for tr, te in cv.split(x, intervals)]
    assert a == b


# --- PurgedKFold: purging correctness (the core G16 invariant) ---------------


@pytest.mark.parametrize("horizon", [1, 3, 7, 15])
@pytest.mark.parametrize("n_splits", [2, 3, 5])
def test_purged_kfold_no_leakage_across_horizons(horizon: int, n_splits: int) -> None:
    n = 120
    intervals = _fixed_horizon_intervals(n, horizon=horizon)
    x = np.zeros(n)
    cv = PurgedKFold(n_splits=n_splits, embargo_frac=0.0)
    for train_idx, test_idx in cv.split(x, intervals):
        _assert_no_train_test_overlap(train_idx, test_idx, intervals)
        # train and test never share a position
        assert set(train_idx).isdisjoint(set(test_idx.tolist()))


def test_purged_kfold_purges_all_three_overlap_cases() -> None:
    """t0-inside, t1-inside and envelopment cases are all purged (AFML §7)."""
    # Hand-built intervals; positions are the index, values are label ends.
    t0 = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    t1 = np.array([10, 3, 4, 6, 5, 6, 8, 9, 10, 11])  # sample 0 envelops the middle
    intervals = pd.Series(t1, index=t0)
    x = np.zeros(len(t0))
    cv = PurgedKFold(n_splits=2, embargo_frac=0.0)
    for train_idx, test_idx in cv.split(x, intervals):
        _assert_no_train_test_overlap(train_idx, test_idx, intervals)


def test_purged_kfold_embargo_removes_trailing_fraction() -> None:
    n = 100
    embargo_frac = 0.1
    intervals = _fixed_horizon_intervals(n, horizon=0)  # zero horizon isolates embargo
    x = np.zeros(n)
    cv = PurgedKFold(n_splits=5, embargo_frac=embargo_frac)
    embargo_n = int(n * embargo_frac)
    for train_idx, test_idx in cv.split(x, intervals):
        last_test = int(test_idx[-1])
        banned = set(range(last_test + 1, last_test + 1 + embargo_n))
        assert banned.isdisjoint(set(train_idx.tolist())), "embargoed positions leaked into train"


def test_purged_kfold_zero_embargo_still_purges() -> None:
    """embargo_frac=0 must NOT reduce to plain k-fold — purging is unconditional."""
    n = 60
    intervals = _fixed_horizon_intervals(n, horizon=6)
    x = np.zeros(n)
    cv = PurgedKFold(n_splits=3, embargo_frac=0.0)
    for train_idx, test_idx in cv.split(x, intervals):
        _assert_no_train_test_overlap(train_idx, test_idx, intervals)
        # With overlap, purging must have dropped some otherwise-eligible samples.
    # A middle fold: some non-test positions are dropped by purging.
    middle = list(PurgedKFold(n_splits=3, embargo_frac=0.0).split(x, intervals))[1]
    train_idx, test_idx = middle
    non_test = set(range(n)) - set(test_idx.tolist())
    assert len(train_idx) < len(non_test)  # purging removed boundary samples


def test_purged_kfold_no_way_to_disable_purging() -> None:
    """G9: an unpurged configuration must be impossible to construct."""
    cv = PurgedKFold(n_splits=3, embargo_frac=0.0)
    # No public flag toggles purging off.
    assert not hasattr(cv, "purge")
    assert not hasattr(cv, "no_purge")


# --- PurgedKFold: validation -------------------------------------------------


def test_purged_kfold_length_mismatch_raises() -> None:
    intervals = _fixed_horizon_intervals(50, horizon=2)
    with pytest.raises(ValueError, match="length"):
        list(PurgedKFold(n_splits=3, embargo_frac=0.0).split(np.zeros(40), intervals))


@pytest.mark.parametrize("bad", [0, 1, -1])
def test_purged_kfold_rejects_bad_n_splits(bad: int) -> None:
    with pytest.raises(ValueError):
        PurgedKFold(n_splits=bad, embargo_frac=0.0)


@pytest.mark.parametrize("bad", [-0.1, 1.0, 1.5])
def test_purged_kfold_rejects_bad_embargo(bad: float) -> None:
    with pytest.raises(ValueError):
        PurgedKFold(n_splits=3, embargo_frac=bad)


def test_purged_kfold_rejects_non_monotonic_t0() -> None:
    intervals = pd.Series([5, 6, 7], index=[2, 0, 1])  # t0 out of order
    with pytest.raises(ValueError, match="monotonic"):
        list(PurgedKFold(n_splits=2, embargo_frac=0.0).split(np.zeros(3), intervals))


def test_purged_kfold_rejects_t1_before_t0() -> None:
    intervals = pd.Series([0, 1, 2], index=[0, 1, 5])  # t1[2]=2 < t0[2]=5
    with pytest.raises(ValueError, match="t1"):
        list(PurgedKFold(n_splits=2, embargo_frac=0.0).split(np.zeros(3), intervals))


def test_purged_kfold_supports_datetime_intervals() -> None:
    n = 40
    t0 = pd.date_range("2020-01-01", periods=n, freq="D")
    t1 = t0 + pd.Timedelta(days=3)
    intervals = pd.Series(t1, index=t0)
    x = np.zeros(n)
    for train_idx, test_idx in PurgedKFold(n_splits=4, embargo_frac=0.0).split(x, intervals):
        _assert_no_train_test_overlap(train_idx, test_idx, intervals)


# --- CombinatorialPurgedCV: exact combinatorial structure --------------------


def test_cpcv_produces_all_combinations_and_paths() -> None:
    """Hand-verified small case: N=6, k=2 → C(6,2)=15 splits, 5 backtest paths."""
    n = 30
    intervals = _fixed_horizon_intervals(n, horizon=1)
    x = np.zeros(n)
    cv = CombinatorialPurgedCV(n_groups=6, n_test_groups=2, embargo_frac=0.0)
    assert cv.n_splits == 15
    assert cv.n_paths == 5  # phi = k * C(N,k) / N = 2*15/6
    splits = list(cv.split(x, intervals))
    assert len(splits) == 15


def test_cpcv_group_assignments_match_all_combinations() -> None:
    n = 30
    n_groups, n_test = 6, 2
    intervals = _fixed_horizon_intervals(n, horizon=1)
    x = np.zeros(n)
    groups = np.array_split(np.arange(n), n_groups)
    pos_to_group = {int(p): g for g, arr in enumerate(groups) for p in arr}

    cv = CombinatorialPurgedCV(n_groups=n_groups, n_test_groups=n_test, embargo_frac=0.0)
    seen_combos: set[tuple[int, ...]] = set()
    for _, test_idx in cv.split(x, intervals):
        combo = tuple(sorted({pos_to_group[int(p)] for p in test_idx}))
        assert len(combo) == n_test  # exactly k groups form each test set
        seen_combos.add(combo)
    assert seen_combos == set(itertools.combinations(range(n_groups), n_test))


@pytest.mark.parametrize("horizon", [1, 4, 9])
def test_cpcv_no_leakage_with_disjoint_test_groups(horizon: int) -> None:
    n = 60
    intervals = _fixed_horizon_intervals(n, horizon=horizon)
    x = np.zeros(n)
    cv = CombinatorialPurgedCV(n_groups=6, n_test_groups=2, embargo_frac=0.05)
    for train_idx, test_idx in cv.split(x, intervals):
        # test groups may be non-contiguous; invariant must still hold everywhere.
        _assert_no_train_test_overlap(train_idx, test_idx, intervals)


@pytest.mark.parametrize(
    ("n_groups", "n_test"),
    [(1, 1), (3, 0), (3, 3), (3, 4)],
)
def test_cpcv_rejects_bad_group_config(n_groups: int, n_test: int) -> None:
    with pytest.raises(ValueError):
        CombinatorialPurgedCV(n_groups=n_groups, n_test_groups=n_test, embargo_frac=0.0)


def test_cpcv_n_paths_formula_various() -> None:
    for n_groups, n_test in [(6, 2), (10, 2), (8, 4)]:
        cv = CombinatorialPurgedCV(n_groups=n_groups, n_test_groups=n_test, embargo_frac=0.0)
        assert cv.n_splits == math.comb(n_groups, n_test)
        assert cv.n_paths == n_test * math.comb(n_groups, n_test) // n_groups


# --- CPCV cross-check against skfolio (dev-only) -----------------------------


def test_cpcv_matches_skfolio_on_index_purge_compatible_case() -> None:
    """Independent cross-check: fold structure agrees with skfolio's CPCV.

    We use a fixed-horizon (constant span) label set so that label-interval
    purging coincides with skfolio's index-based ``purged_size``. Test-fold
    assignments must match exactly; training sets must match where the two
    purging semantics align (documented: our label-interval purge is never
    laxer than skfolio's index purge).
    """
    skfolio_ms = pytest.importorskip("skfolio.model_selection")
    n, n_groups, n_test, horizon = 60, 6, 2, 2
    intervals = _fixed_horizon_intervals(n, horizon=horizon)
    x = np.zeros((n, 1))

    ours = CombinatorialPurgedCV(n_groups=n_groups, n_test_groups=n_test, embargo_frac=0.0)
    their_cv = skfolio_ms.CombinatorialPurgedCV(
        n_folds=n_groups, n_test_folds=n_test, purged_size=horizon, embargo_size=0
    )
    assert ours.n_splits == their_cv.n_splits
    assert ours.n_paths == their_cv.n_test_paths

    def _test_set(test_obj: object) -> frozenset[int]:
        if isinstance(test_obj, (list, tuple)):
            return frozenset(int(p) for arr in test_obj for p in np.asarray(arr))
        return frozenset(int(p) for p in np.asarray(test_obj))

    our_by_test = {
        frozenset(int(p) for p in te): frozenset(int(p) for p in tr)
        for tr, te in ours.split(x, intervals)
    }
    their_by_test = {
        _test_set(te): frozenset(int(p) for p in np.asarray(tr)) for tr, te in their_cv.split(x)
    }
    # Test-fold assignments match exactly.
    assert set(our_by_test) == set(their_by_test)
    # Our training set is a subset of skfolio's (never laxer) for every split.
    for test_set, our_train in our_by_test.items():
        assert our_train <= their_by_test[test_set]


# --- WalkForward: structure and purging --------------------------------------


def test_walk_forward_expanding_train_and_contiguous_test() -> None:
    n, folds, min_train = 100, 4, 20
    intervals = _fixed_horizon_intervals(n, horizon=0)
    x = np.zeros(n)
    splits = list(WalkForward(n_folds=folds, min_train=min_train).split(x, intervals))
    assert len(splits) == folds
    prev_train_end = -1
    for train_idx, test_idx in splits:
        assert list(test_idx) == list(range(int(test_idx[0]), int(test_idx[-1]) + 1))
        assert int(test_idx[0]) >= min_train
        assert int(train_idx.max()) < int(test_idx[0])  # strictly walk-forward
        assert int(test_idx[0]) > prev_train_end
        prev_train_end = int(test_idx[-1])


def test_walk_forward_purge_truncates_leaking_labels() -> None:
    n, folds, min_train, horizon = 100, 4, 20, 5
    intervals = _fixed_horizon_intervals(n, horizon=horizon)
    x = np.zeros(n)
    for train_idx, test_idx in WalkForward(n_folds=folds, min_train=min_train).split(x, intervals):
        _assert_no_train_test_overlap(train_idx, test_idx, intervals)


def test_walk_forward_public_api_cannot_disable_purge() -> None:
    """G9 / architect note 5: the unpurged path is never reachable publicly."""
    with pytest.raises(ValueError, match="purg"):
        WalkForward(n_folds=3, min_train=10, purge=False)


def test_walk_forward_validates_configuration() -> None:
    with pytest.raises(ValueError):
        WalkForward(n_folds=0, min_train=10)
    with pytest.raises(ValueError):
        WalkForward(n_folds=3, min_train=-1)
    intervals = _fixed_horizon_intervals(10, horizon=1)
    with pytest.raises(ValueError):
        # min_train leaves fewer than n_folds test observations
        list(WalkForward(n_folds=8, min_train=8).split(np.zeros(10), intervals))


# --- Leakage regression (AUDIT §2.5) -----------------------------------------


def test_leakage_regression_purge_lowers_inflated_skill() -> None:
    """Reproduces the legacy boundary-leak bug and proves purging removes it.

    Overlapping horizon-h labels y_i = sum(shocks[i:i+h]) are autocorrelated:
    corr(y_i, y_{i+d}) = (h-d)/h for d<h. A persistence forecaster predicts each
    test label from the freshest *usable* training label. Without purging the
    freshest training label overlaps the test window (lag 1, corr (h-1)/h ≈ 0.9),
    inflating out-of-sample skill. Purging forces the freshest usable label back
    by ~h observations (lag > h, corr ≈ 0), collapsing the spurious skill. The
    unpurged path is reachable ONLY through the private test hook below — never
    public API — so plain (leaky) CV stays unconstructable in production.
    """
    rng = np.random.default_rng(20260725)
    n, folds, min_train, horizon = 300, 5, 100, 20
    shocks = rng.normal(size=n + horizon)
    y = np.array([shocks[i : i + horizon].sum() for i in range(n)])
    intervals = _fixed_horizon_intervals(n, horizon=horizon)
    x = np.zeros(n)

    def _persistence_skill(splits: list[tuple[np.ndarray, np.ndarray]]) -> float:
        preds: list[float] = []
        actuals: list[float] = []
        for train_idx, test_idx in splits:
            if train_idx.size == 0:
                continue
            last_train = int(train_idx.max())  # freshest usable training label
            gap = int(test_idx[0]) - last_train  # 1 if leaking, >h if purged
            for j in test_idx:
                src = int(j) - gap
                if 0 <= src < n:
                    preds.append(y[src])
                    actuals.append(y[int(j)])
        if len(preds) < 3:
            return 0.0
        c = np.corrcoef(preds, actuals)[0, 1]
        return 0.0 if np.isnan(c) else float(c)

    wf = WalkForward(n_folds=folds, min_train=min_train)
    purged_skill = _persistence_skill(list(wf.split(x, intervals)))
    # Private, test-only hook exposes the unpurged (leaky) splits.
    unpurged_skill = _persistence_skill(list(wf._iter_splits(x, intervals, purge=False)))

    assert unpurged_skill > 0.5  # leak inflates apparent skill
    assert purged_skill < 0.25  # purging collapses it to ~chance
    assert unpurged_skill - purged_skill > 0.3  # measurable, sizeable gap
