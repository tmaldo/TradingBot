"""Leak-proof, label-interval-aware cross-validation splitters (G9 / G16).

Implements the three cross-validation schemes mandated by Global Constraint G9,
following López de Prado, *Advances in Financial Machine Learning* (AFML) §7
(purging & embargo) and §12 (Combinatorial Purged Cross-Validation):

* :class:`PurgedKFold` -- purged k-fold with embargo.
* :class:`CombinatorialPurgedCV` -- CPCV over all ``C(n_groups, n_test_groups)``
  train/test partitions.
* :class:`WalkForward` -- expanding-window walk-forward with boundary purging.

**Plain k-fold is forbidden and unconstructable.** Purging is unconditional in
every splitter: there is no public flag, argument, or subclass hook that turns
it off. ``WalkForward`` accepts ``purge`` only to *reject* ``purge=False`` at
construction; the unpurged code path exists solely as a private method used by
the leakage regression test (AUDIT §2.5) and is never reachable through the
public API.

Label intervals (De Prado's ``(t0, t1)`` spans)
------------------------------------------------
A :data:`LabelIntervals` is a ``pandas.Series`` whose *index* holds each
sample's event-start time ``t0`` and whose *values* hold the corresponding
label-end time ``t1``. It aligns to the rows of ``X`` **positionally** (row ``i``
of ``X`` ↔ the ``i``-th entry of ``intervals``); a length mismatch raises.

Purging semantics (AFML §7)
---------------------------
A training sample ``i`` with label interval ``[t0_i, t1_i]`` is dropped iff that
interval overlaps the interval of *any* test sample ``j`` (``[t0_j, t1_j]``).
Closed-interval overlap holds iff ``t0_i <= t1_j and t0_j <= t1_i``; this single
condition subsumes De Prado's three enumerated cases (test ``t0`` inside the
train span, test ``t1`` inside it, and full envelopment). Because each test fold
is a *contiguous block of positions*, purging against the block's envelope
``[min t0_test, max t1_test]`` is exact — every training sample lies entirely
before or after the block, so envelope-overlap coincides with any-interval
overlap and never under-purges. CPCV, whose test set is a union of possibly
non-contiguous groups, purges against each selected group's envelope in turn.

Embargo (AFML §7)
-----------------
Beyond purging, an embargo additionally drops training samples whose position
falls within ``embargo_frac * n_obs`` observations immediately **after** a test
block's end, guarding against leakage through serial correlation that outlives
the label window.

The splitters are deterministic (no RNG) and vectorised.
"""

from __future__ import annotations

import math
from collections.abc import Iterator
from itertools import combinations

import numpy as np
import numpy.typing as npt
import pandas as pd

# index = t0 (event start), values = t1 (label end); aligns to X rows by position.
LabelIntervals = pd.Series

IntArray = npt.NDArray[np.intp]
Split = tuple[IntArray, IntArray]


def _validate_intervals(x: object, intervals: LabelIntervals) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(t0, t1)`` numpy arrays after validating shape and ordering."""
    n_x = len(x)  # type: ignore[arg-type]
    if len(intervals) != n_x:
        raise ValueError(f"intervals length ({len(intervals)}) must match X length ({n_x})")
    t0 = intervals.index.to_numpy()
    t1 = intervals.to_numpy()
    if n_x == 0:
        raise ValueError("cannot split an empty dataset")
    # t0 must be chronological so positional embargo windows are meaningful.
    if np.any(t0[1:] < t0[:-1]):
        raise ValueError("intervals index (t0) must be monotonic non-decreasing")
    if np.any(t1 < t0):
        raise ValueError("each label end t1 must be >= its start t0")
    return t0, t1


def _purge_and_embargo_mask(
    t0: np.ndarray,
    t1: np.ndarray,
    test_blocks: list[IntArray],
    embargo_n: int,
) -> npt.NDArray[np.bool_]:
    """Boolean mask of samples removed by purging + embargo against ``test_blocks``.

    Each block is a contiguous run of positions. A sample is purged if its label
    interval overlaps the block's envelope ``[min t0, max t1]``; it is embargoed
    if its position lies in ``(block_end, block_end + embargo_n]``.
    """
    n = t0.shape[0]
    removed = np.zeros(n, dtype=bool)
    for block in test_blocks:
        if block.size == 0:
            continue
        block_t0_min = t0[block].min()
        block_t1_max = t1[block].max()
        # Purge: overlap with the block envelope (exact for a contiguous block).
        removed |= (t0 <= block_t1_max) & (t1 >= block_t0_min)
        # Embargo: positions immediately after the block end.
        if embargo_n > 0:
            end = int(block[-1])
            removed[end + 1 : min(n, end + 1 + embargo_n)] = True
    return removed


class PurgedKFold:
    """Purged k-fold cross-validation with embargo (AFML §7, snippet 7.3).

    Observations are partitioned into ``n_splits`` contiguous test folds (in
    position order — never shuffled). For each fold the training set is every
    other sample minus those purged (label overlaps the test block) and embargoed
    (positions just after the test block). Purging is unconditional: there is no
    way to obtain a plain (unpurged) k-fold.

    Parameters
    ----------
    n_splits:
        Number of folds; must be >= 2.
    embargo_frac:
        Fraction of ``n_obs`` embargoed after each test block; in ``[0, 1)``.
        ``0`` disables the *embargo* only — purging still applies.
    """

    def __init__(self, n_splits: int, embargo_frac: float) -> None:
        if n_splits < 2:
            raise ValueError(f"n_splits must be >= 2, got {n_splits}")
        if not (0.0 <= embargo_frac < 1.0):
            raise ValueError(f"embargo_frac must be in [0, 1), got {embargo_frac}")
        self.n_splits = n_splits
        self.embargo_frac = embargo_frac

    def split(self, X: object, intervals: LabelIntervals) -> Iterator[Split]:  # noqa: N803
        """Yield ``(train_idx, test_idx)`` positional index arrays per fold."""
        t0, t1 = _validate_intervals(X, intervals)
        n = t0.shape[0]
        if self.n_splits > n:
            raise ValueError(f"n_splits ({self.n_splits}) exceeds n_obs ({n})")
        indices = np.arange(n, dtype=np.intp)
        embargo_n = int(n * self.embargo_frac)
        for test_idx in np.array_split(indices, self.n_splits):
            test_idx = np.asarray(test_idx, dtype=np.intp)
            removed = _purge_and_embargo_mask(t0, t1, [test_idx], embargo_n)
            train_mask = np.ones(n, dtype=bool)
            train_mask[test_idx] = False
            train_mask &= ~removed
            yield indices[train_mask], test_idx


class CombinatorialPurgedCV:
    """Combinatorial Purged Cross-Validation (AFML §12).

    Observations are partitioned into ``n_groups`` contiguous groups. Every
    ``C(n_groups, n_test_groups)`` choice of test groups yields one split whose
    test set is the union of those groups and whose training set is everything
    else, purged and embargoed against each selected group. This produces
    ``n_paths = n_test_groups * C(n_groups, n_test_groups) / n_groups`` distinct
    backtest paths (López de Prado, AFML eq. 12.1).

    Parameters
    ----------
    n_groups:
        Number of contiguous groups; must be >= 2.
    n_test_groups:
        Groups assigned to the test set per split; ``1 <= n_test_groups < n_groups``.
    embargo_frac:
        Embargo fraction of ``n_obs`` applied after each selected test group.
    """

    def __init__(self, n_groups: int, n_test_groups: int, embargo_frac: float) -> None:
        if n_groups < 2:
            raise ValueError(f"n_groups must be >= 2, got {n_groups}")
        if not (1 <= n_test_groups < n_groups):
            raise ValueError(f"n_test_groups must satisfy 1 <= k < n_groups, got {n_test_groups}")
        if not (0.0 <= embargo_frac < 1.0):
            raise ValueError(f"embargo_frac must be in [0, 1), got {embargo_frac}")
        self.n_groups = n_groups
        self.n_test_groups = n_test_groups
        self.embargo_frac = embargo_frac

    @property
    def n_splits(self) -> int:
        """Number of train/test partitions = ``C(n_groups, n_test_groups)``."""
        return math.comb(self.n_groups, self.n_test_groups)

    @property
    def n_paths(self) -> int:
        """Number of recombinable backtest paths = ``k * C(N, k) / N``."""
        return self.n_test_groups * self.n_splits // self.n_groups

    def split(self, X: object, intervals: LabelIntervals) -> Iterator[Split]:  # noqa: N803
        """Yield ``(train_idx, test_idx)`` for every test-group combination."""
        t0, t1 = _validate_intervals(X, intervals)
        n = t0.shape[0]
        if self.n_groups > n:
            raise ValueError(f"n_groups ({self.n_groups}) exceeds n_obs ({n})")
        indices = np.arange(n, dtype=np.intp)
        groups = [np.asarray(g, dtype=np.intp) for g in np.array_split(indices, self.n_groups)]
        embargo_n = int(n * self.embargo_frac)
        for combo in combinations(range(self.n_groups), self.n_test_groups):
            test_groups = [groups[g] for g in combo]
            test_idx = np.concatenate(test_groups)
            test_idx.sort()
            removed = _purge_and_embargo_mask(t0, t1, test_groups, embargo_n)
            train_mask = np.ones(n, dtype=bool)
            train_mask[test_idx] = False
            train_mask &= ~removed
            yield indices[train_mask], test_idx


class WalkForward:
    """Expanding-window walk-forward validation with boundary purging (AFML §7).

    The observations after ``min_train`` are divided into ``n_folds`` contiguous
    test blocks. Fold ``f`` trains on every observation before its test block and
    (with ``purge=True``) drops trailing training samples whose label interval
    extends into the test window — exactly the boundary leak the legacy expanding
    walk-forward suffered (AUDIT §2.5).

    ``purge`` exists only to enforce that it is never ``False``: constructing an
    unpurged walk-forward raises, keeping plain (leaky) CV unconstructable in
    production (G9 / architect note 5). The leakage regression test reaches the
    unpurged path via the private :meth:`_iter_splits` hook, never public API.

    Parameters
    ----------
    n_folds:
        Number of test blocks; must be >= 1.
    min_train:
        Minimum training observations before the first test block; must be >= 0
        and leave at least ``n_folds`` observations for testing.
    purge:
        Must be ``True`` (the default). ``purge=False`` is rejected.
    """

    def __init__(self, n_folds: int, min_train: int, purge: bool = True) -> None:
        if not purge:
            raise ValueError(
                "unpurged walk-forward is forbidden (G9): purging is unconditional. "
                "The leaky path exists only for the AUDIT §2.5 regression test."
            )
        if n_folds < 1:
            raise ValueError(f"n_folds must be >= 1, got {n_folds}")
        if min_train < 0:
            raise ValueError(f"min_train must be >= 0, got {min_train}")
        self.n_folds = n_folds
        self.min_train = min_train

    def split(self, X: object, intervals: LabelIntervals) -> Iterator[Split]:  # noqa: N803
        """Yield ``(train_idx, test_idx)`` per fold (always purged)."""
        yield from self._iter_splits(X, intervals, purge=True)

    def _iter_splits(
        self,
        X: object,  # noqa: N803
        intervals: LabelIntervals,
        purge: bool,
    ) -> Iterator[Split]:
        """Core split logic. ``purge=False`` is the leaky path — TEST ONLY.

        This is deliberately private. The public :meth:`split` always passes
        ``purge=True``; the only caller of ``purge=False`` is the leakage
        regression test, which documents the legacy bug.
        """
        t0, t1 = _validate_intervals(X, intervals)
        n = t0.shape[0]
        if self.min_train >= n:
            raise ValueError(f"min_train ({self.min_train}) must be < n_obs ({n})")
        test_region = np.arange(self.min_train, n, dtype=np.intp)
        if test_region.size < self.n_folds:
            raise ValueError(
                f"only {test_region.size} observations after min_train "
                f"({self.min_train}) but n_folds={self.n_folds}"
            )
        for test_idx in np.array_split(test_region, self.n_folds):
            test_idx = np.asarray(test_idx, dtype=np.intp)
            test_start = int(test_idx[0])
            train_idx = np.arange(test_start, dtype=np.intp)
            if purge:
                # Drop trailing train samples whose label reaches the test window.
                test_t0_min = t0[test_idx].min()
                train_idx = train_idx[t1[train_idx] < test_t0_min]
            yield train_idx, test_idx
