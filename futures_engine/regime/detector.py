"""Market-regime detectors with a strictly point-in-time labeling path (G4/G6).

A regime module lets strategies be *gated* by market state (e.g. trend vs.
chop, calm vs. turbulent). The hard constraint is causality: the regime label at
time ``t`` may use only information available at ``t`` -- otherwise every
downstream backtest is silently inflated (G4). Concretely:

* ``fit(bars)`` may use the whole history -- this is research-time parameter
  estimation and is *not* on the decision path.
* ``regimes(bars)`` / ``proba(bars)`` must be **causal**: the value at bar ``t``
  is a function only of bars ``<= t`` and the frozen fitted parameters.

Why no hmmlearn
---------------
The plan named ``hmmlearn`` for the HMM, but (a) it ships no Python 3.14 wheel
(the interpreter this repo runs on) and (b) its public ``predict``/``decode`` API
smooths over the *entire* sequence (Viterbi / posterior decoding), which is
forbidden here. Per the plan's tech-stack note, a self-contained Gaussian HMM is
used instead: parameters are estimated with Baum-Welch in :meth:`fit`, but the
labeling path runs a **forward filter** (``P(state_t | obs_1..obs_t)``), which is
causal by construction. (``statsmodels`` Markov-switching -- also installed -- is
an equally acceptable substitute and offers the same filtered-vs-smoothed split;
the in-house filter is chosen so the causal path is fully under our control and
carries no non-3.14 dependency.)

``ChangePointRegimeDetector`` wraps ``ruptures`` in an **expanding-window**
(walk-forward) loop: the regime id at ``t`` is the index of the current segment
detected using only bars ``<= t``. Both detectors are deterministic and their
tuning parameters are explicit constructor arguments (no magic constants).
"""

from __future__ import annotations

import math
from typing import Protocol, Self, runtime_checkable

import numpy as np
import numpy.typing as npt
import pandas as pd
import ruptures as rpt

from futures_engine.core.types import Bars
from futures_engine.data.audit import reference_history, register_pit_check, registered_checks

FloatArray = npt.NDArray[np.float64]

# Sentinel regime for bars with no return-based observation yet (e.g. the very
# first bar). Distinct from any real state id (which are 0..n_states-1).
WARMUP_REGIME = -1


@runtime_checkable
class RegimeDetector(Protocol):
    """Structural interface every regime detector satisfies."""

    def fit(self, bars: Bars) -> Self:
        """Estimate parameters (research-time; may use the whole history)."""

    def regimes(self, bars: Bars) -> pd.Series:
        """Causal integer regime label per bar (value at t uses bars <= t)."""

    def proba(self, bars: Bars) -> pd.DataFrame:
        """Causal per-regime probability per bar (rows = bars, cols = regimes)."""


def _log_returns(bars: Bars) -> FloatArray:
    close = bars["close"].to_numpy(dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        rets = np.diff(np.log(close))
    return rets.astype(np.float64)


class HMMRegimeDetector:
    """Self-contained Gaussian HMM on log-returns; causal forward-filter labeling.

    Parameters
    ----------
    n_states:
        Number of hidden regimes (``>= 2``).
    seed:
        RNG seed for the (deterministic) Baum-Welch initialization.
    max_iter, tol:
        EM stopping controls (documented defaults, not magic constants).
    min_variance:
        Floor on each state's emission variance, preventing a degenerate state
        from collapsing to zero variance (numerical safety).
    """

    def __init__(
        self,
        n_states: int,
        seed: int,
        *,
        max_iter: int = 50,
        tol: float = 1e-4,
        min_variance: float = 1e-8,
    ) -> None:
        if n_states < 2:
            raise ValueError(f"n_states must be >= 2, got {n_states}")
        if max_iter < 1:
            raise ValueError(f"max_iter must be >= 1, got {max_iter}")
        if min_variance <= 0:
            raise ValueError(f"min_variance must be > 0, got {min_variance}")
        self.n_states = n_states
        self.seed = seed
        self.max_iter = max_iter
        self.tol = tol
        self.min_variance = min_variance
        self.startprob_: FloatArray | None = None
        self.transmat_: FloatArray | None = None
        self.means_: FloatArray | None = None
        self.variances_: FloatArray | None = None

    # -- emissions ------------------------------------------------------------

    def _emission(self, obs: FloatArray) -> FloatArray:
        """Gaussian emission likelihood matrix ``B[t, k]`` (shape ``(T, K)``)."""
        assert self.means_ is not None and self.variances_ is not None
        diff = obs[:, None] - self.means_[None, :]
        coeff = 1.0 / np.sqrt(2.0 * math.pi * self.variances_)
        expo = np.exp(-0.5 * diff * diff / self.variances_)
        out: FloatArray = coeff[None, :] * expo
        return out

    # -- fit (Baum-Welch; full history allowed) -------------------------------

    def fit(self, bars: Bars) -> Self:
        obs = _log_returns(bars)
        obs = obs[np.isfinite(obs)]
        k = self.n_states
        rng = np.random.default_rng(self.seed)
        if obs.size < k:
            # Degenerate: not enough data to estimate; fall back to a flat model.
            self.startprob_ = np.full(k, 1.0 / k)
            self.transmat_ = np.full((k, k), 1.0 / k)
            self.means_ = np.linspace(-1.0, 1.0, k)
            self.variances_ = np.full(k, max(self.min_variance, 1.0))
            return self

        quantiles = np.quantile(obs, np.linspace(0.0, 1.0, k + 2)[1:-1])
        self.means_ = quantiles + rng.normal(0.0, 1e-6, k)
        self.variances_ = np.full(k, max(float(np.var(obs)), self.min_variance))
        self.startprob_ = np.full(k, 1.0 / k)
        transmat = np.full((k, k), 1.0 / k) + rng.normal(0.0, 1e-3, (k, k))
        transmat = np.abs(transmat)
        self.transmat_ = transmat / transmat.sum(axis=1, keepdims=True)

        prev_ll = -np.inf
        for _ in range(self.max_iter):
            emission = self._emission(obs)
            alpha, scale = self._forward(emission)
            beta = self._backward(emission, scale)
            gamma = alpha * beta
            gamma /= gamma.sum(axis=1, keepdims=True)
            xi = self._xi(emission, alpha, beta)

            self.startprob_ = gamma[0]
            trans = xi.sum(axis=0)
            self.transmat_ = trans / trans.sum(axis=1, keepdims=True)
            weight = gamma.sum(axis=0)
            self.means_ = (gamma * obs[:, None]).sum(axis=0) / weight
            resid = obs[:, None] - self.means_[None, :]
            var = (gamma * resid * resid).sum(axis=0) / weight
            self.variances_ = np.maximum(var, self.min_variance)

            log_likelihood = float(np.log(scale).sum())
            if abs(log_likelihood - prev_ll) < self.tol:
                break
            prev_ll = log_likelihood
        return self

    def _forward(self, emission: FloatArray) -> tuple[FloatArray, FloatArray]:
        assert self.startprob_ is not None and self.transmat_ is not None
        t_len = emission.shape[0]
        alpha = np.zeros_like(emission)
        scale = np.zeros(t_len)
        a0 = self.startprob_ * emission[0]
        scale[0] = a0.sum() or 1.0
        alpha[0] = a0 / scale[0]
        for t in range(1, t_len):
            a = emission[t] * (alpha[t - 1] @ self.transmat_)
            scale[t] = a.sum() or 1.0
            alpha[t] = a / scale[t]
        return alpha, scale

    def _backward(self, emission: FloatArray, scale: FloatArray) -> FloatArray:
        assert self.transmat_ is not None
        t_len = emission.shape[0]
        beta = np.zeros_like(emission)
        beta[-1] = 1.0
        for t in range(t_len - 2, -1, -1):
            beta[t] = (self.transmat_ @ (emission[t + 1] * beta[t + 1])) / scale[t + 1]
        return beta

    def _xi(self, emission: FloatArray, alpha: FloatArray, beta: FloatArray) -> FloatArray:
        assert self.transmat_ is not None
        t_len = emission.shape[0]
        k = self.n_states
        xi = np.zeros((t_len - 1, k, k))
        for t in range(t_len - 1):
            num = (
                alpha[t][:, None] * self.transmat_ * emission[t + 1][None, :] * beta[t + 1][None, :]
            )
            total = num.sum()
            xi[t] = num / (total or 1.0)
        return xi

    # -- causal labeling (forward filter only) --------------------------------

    def _filtered(self, bars: Bars) -> tuple[pd.Index, FloatArray]:
        """Filtered posteriors ``P(state_t | obs_1..obs_t)`` aligned to bars 1..n-1."""
        assert self.startprob_ is not None and self.transmat_ is not None
        obs = _log_returns(bars)
        t_len = obs.shape[0]
        k = self.n_states
        filt = np.full((t_len, k), np.nan, dtype=np.float64)
        prev: FloatArray | None = None
        for t in range(t_len):
            if not np.isfinite(obs[t]):
                prev = None
                continue
            emission = self._emission(obs[t : t + 1])[0]
            pred = self.startprob_ if prev is None else prev @ self.transmat_
            a = emission * pred
            total = a.sum()
            a = a / total if total > 0 else np.full(k, 1.0 / k)
            filt[t] = a
            prev = a
        return bars.index, filt

    def regimes(self, bars: Bars) -> pd.Series:
        idx, filt = self._filtered(bars)
        out = np.full(len(idx), WARMUP_REGIME, dtype=np.int64)
        # filt row j corresponds to the return at bar j+1.
        valid = ~np.isnan(filt).any(axis=1)
        labels = np.argmax(filt[valid], axis=1).astype(np.int64)
        out[1:][valid] = labels
        return pd.Series(out, index=idx)

    def proba(self, bars: Bars) -> pd.DataFrame:
        idx, filt = self._filtered(bars)
        mat = np.full((len(idx), self.n_states), np.nan)
        mat[1:] = filt
        return pd.DataFrame(mat, index=idx, columns=list(range(self.n_states)))


class ChangePointRegimeDetector:
    """Expanding-window (walk-forward) change-point regimes via ``ruptures``.

    The regime id at bar ``t`` is the index of the current segment detected by
    running the offline algorithm on bars ``<= t`` only -- so the label is causal
    even though ``ruptures`` itself is an offline detector. ``model`` and
    ``penalty`` are the ``ruptures`` cost model and penalty; ``min_size``/``jump``
    are its resolution controls.
    """

    def __init__(
        self,
        model: str,
        penalty: float,
        *,
        min_size: int = 2,
        jump: int = 1,
    ) -> None:
        if not model:
            raise ValueError("model must be a non-empty ruptures cost model name")
        if penalty <= 0:
            raise ValueError(f"penalty must be > 0, got {penalty}")
        if min_size < 1:
            raise ValueError(f"min_size must be >= 1, got {min_size}")
        if jump < 1:
            raise ValueError(f"jump must be >= 1, got {jump}")
        self.model = model
        self.penalty = penalty
        self.min_size = min_size
        self.jump = jump
        self.n_regimes_ = 1

    def _signal(self, bars: Bars) -> FloatArray:
        return bars["close"].to_numpy(dtype=np.float64)

    def _n_segments(self, signal: FloatArray) -> int:
        if signal.shape[0] < 2 * self.min_size:
            return 1
        algo = rpt.Pelt(model=self.model, min_size=self.min_size, jump=self.jump)
        result = algo.fit(signal.reshape(-1, 1)).predict(pen=self.penalty)
        return len(result)

    def fit(self, bars: Bars) -> Self:
        self.n_regimes_ = self._n_segments(self._signal(bars))
        return self

    def regimes(self, bars: Bars) -> pd.Series:
        signal = self._signal(bars)
        n = signal.shape[0]
        out = np.zeros(n, dtype=np.int64)
        for t in range(n):
            out[t] = self._n_segments(signal[: t + 1]) - 1
        return pd.Series(out, index=bars.index)

    def proba(self, bars: Bars) -> pd.DataFrame:
        reg = self.regimes(bars)
        k = max(self.n_regimes_, int(reg.max()) + 1)
        mat = np.zeros((len(reg), k), dtype=np.float64)
        mat[np.arange(len(reg)), np.clip(reg.to_numpy(), 0, k - 1)] = 1.0
        return pd.DataFrame(mat, index=bars.index, columns=list(range(k)))


def register_regime_checks(seed: int = 0) -> None:
    """Register the causal regime labeling path with the look-ahead audit (G4).

    A detector is fit once on the deterministic audit reference history
    (research-time), then its causal :meth:`HMMRegimeDetector.regimes` is
    registered so CI shift-tests it. Idempotent.
    """
    if "regime.hmm" in registered_checks():
        return
    detector = HMMRegimeDetector(n_states=2, seed=seed).fit(reference_history())
    register_pit_check("regime.hmm", detector.regimes)
