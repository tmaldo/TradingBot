"""Meta-labeling pipeline: LightGBM primary + L2-logistic baseline (G5/G6).

Meta-labeling (López de Prado, AFML §3.6): a primary :class:`VectorSignal`
proposes a *side* at each triple-barrier event; the meta-model predicts the
*probability that taking that side wins* (``p(trade) in [0, 1]``). Positioning
sizing then follows from the probability rather than from the primary alone.

What this module guarantees
---------------------------
* **Two learners, side by side.** Every :meth:`MetaModelPipeline.fit` trains
  both a LightGBM classifier and an L2-penalised logistic-regression baseline and
  reports both models' out-of-sample metrics in ``FitResult.oos_metrics`` -- so a
  reviewer always sees the gradient-boosted result next to a linear sanity check.
  ``predict`` uses whichever ``model`` the pipeline was constructed with.
* **Uniqueness weights are honoured.** The T4 ``weights`` series is passed
  straight through as ``sample_weight`` to both learners (overlapping labels
  down-weighted, AFML §4).
* **Only the injected splitter decides folds.** OOS evaluation iterates the
  supplied T3 splitter over the events' ``(t0, t1)`` label intervals; there is no
  plain-k-fold code path anywhere -- purging/embargo come for free from T3.
* **Deterministic + logged.** Both learners are seeded; every fit writes exactly
  one :class:`TrialRecord` (the DSR trial count stays honest, G10).

Features
--------
A documented minimal feature set is derived from bars alone so ``predict(bars)``
is self-contained: trailing returns over 1/5/20 bars, 20-bar realised vol, and
the primary signal's raw value. Production would substitute the richer T4 feature
matrix; that integration is intentionally out of scope here (the pipeline is
decoupled from ``futures_engine.features``).
"""

from __future__ import annotations

import hashlib
import json
import tempfile
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score

from futures_engine.core.manifest import current_git_sha
from futures_engine.core.types import Bars
from futures_engine.research.harness import Splitter, VectorSignal, _cv_scheme
from futures_engine.trials.logger import TrialLogger, TrialRecord

ModelName = Literal["lightgbm", "logistic_l2"]
_MODELS: tuple[ModelName, ...] = ("lightgbm", "logistic_l2")
_FEATURE_COLUMNS: tuple[str, ...] = ("ret_1", "ret_5", "ret_20", "vol_20", "primary")
_UNDEFINED = -1.0  # sentinel for a metric that is undefined (e.g. single-class fold)


@dataclass(frozen=True)
class FitResult:
    """Outcome of a meta-model fit.

    ``oos_metrics`` maps each model name to its pooled out-of-sample metrics
    (``accuracy``, ``auc``, ``log_loss``, ``n``, ``pos_rate``); ``per_fold`` has
    one row per (fold, model); ``model_artifact`` is the joblib path of the
    selected, full-data-refit estimator used by :meth:`MetaModelPipeline.predict`.
    """

    oos_metrics: dict[str, dict[str, float]]
    per_fold: pd.DataFrame
    model_artifact: Path


def _accuracy(y_true: np.ndarray, prob: np.ndarray) -> float:
    return float(((prob >= 0.5).astype(int) == y_true).mean())


def _auc(y_true: np.ndarray, prob: np.ndarray) -> float:
    if np.unique(y_true).size < 2:
        return _UNDEFINED
    return float(roc_auc_score(y_true, prob))


def _pooled_metrics(ys: list[np.ndarray], ps: list[np.ndarray]) -> dict[str, float]:
    if not ys:
        return {
            "accuracy": _UNDEFINED,
            "auc": _UNDEFINED,
            "log_loss": _UNDEFINED,
            "n": 0.0,
            "pos_rate": _UNDEFINED,
        }
    y = np.concatenate(ys)
    p = np.concatenate(ps)
    ll = float(log_loss(y, np.clip(p, 1e-7, 1 - 1e-7), labels=[0, 1]))
    return {
        "accuracy": _accuracy(y, p),
        "auc": _auc(y, p),
        "log_loss": ll,
        "n": float(y.size),
        "pos_rate": float(y.mean()),
    }


class MetaModelPipeline:
    """Triple-barrier meta-labeling with a LightGBM/logistic model choice."""

    def __init__(
        self,
        primary: VectorSignal,
        model: ModelName,
        params: dict[str, Any],
        seed: int,
    ) -> None:
        if model not in _MODELS:
            raise ValueError(f"model must be one of {_MODELS}, got {model!r}")
        self.primary = primary
        self.model = model
        self.params = params
        self.seed = seed
        self._fitted: Any = None

    # --- feature + label construction ---------------------------------------

    def _features(self, bars: Bars) -> pd.DataFrame:
        close = bars["close"]
        primary = self.primary.generate(bars, self.params.get("primary", {}))
        return pd.DataFrame(
            {
                "ret_1": close.pct_change(1),
                "ret_5": close.pct_change(5),
                "ret_20": close.pct_change(20),
                "vol_20": close.pct_change().rolling(20).std(),
                "primary": primary.to_numpy(dtype=float),
            },
            index=bars.index,
        )

    def _meta_labels(self, bars: Bars, labels: pd.DataFrame) -> pd.Series:
        """Binary meta-label: 1 iff the primary's side matches a non-zero outcome."""
        side = np.sign(
            self.primary.generate(bars, self.params.get("primary", {}))
            .reindex(labels.index)
            .to_numpy(dtype=float)
        )
        outcome = labels["label"].to_numpy(dtype=float)
        y = ((side == outcome) & (outcome != 0.0)).astype(int)
        return pd.Series(y, index=labels.index)

    # --- estimators ----------------------------------------------------------

    def _make_estimator(self, name: ModelName) -> Any:
        if name == "lightgbm":
            from lightgbm import LGBMClassifier

            return LGBMClassifier(
                random_state=self.seed,
                deterministic=True,
                force_row_wise=True,
                n_jobs=1,
                verbose=-1,
                **dict(self.params.get("lightgbm", {})),
            )
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        # sklearn's default regularisation is L2 (the G6 baseline); we do not pass
        # the deprecated ``penalty`` kwarg -- callers may still override via params.
        clf_params = dict(self.params.get("logistic_l2", {}))
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(random_state=self.seed, **clf_params)),
            ]
        )

    @staticmethod
    def _fit_estimator(
        estimator: Any, name: ModelName, x: np.ndarray, y: np.ndarray, weight: np.ndarray
    ) -> None:
        if name == "lightgbm":
            estimator.fit(x, y, sample_weight=weight)
        else:
            estimator.fit(x, y, clf__sample_weight=weight)

    @staticmethod
    def _prob_positive(estimator: Any, x: np.ndarray) -> np.ndarray:
        proba = np.asarray(estimator.predict_proba(x), dtype=float)
        classes = list(estimator.classes_)
        if 1 in classes:
            return proba[:, classes.index(1)]
        return np.zeros(x.shape[0], dtype=float)

    # --- fit / predict -------------------------------------------------------

    def fit(
        self,
        bars: Bars,
        labels: pd.DataFrame,
        weights: pd.Series,
        splitter: Splitter,
        *,
        logger: TrialLogger,
        snapshot_hashes: list[str] | None = None,
        artifact_dir: str | Path | None = None,
        run_id: str | None = None,
        git_sha: str | None = None,
        ts: datetime | None = None,
    ) -> FitResult:
        """Train + OOS-evaluate both learners; refit the selected one for predict.

        ``logger`` is keyword-only and required: every fit logs one TrialRecord so
        the trial count stays honest. Folds whose training slice is empty or
        single-class (e.g. an extreme walk-forward split) are skipped.
        """
        snapshot_hashes = list(snapshot_hashes) if snapshot_hashes is not None else []

        features = self._features(bars)
        events = features.reindex(labels.index)
        valid = events.notna().all(axis=1)
        events = events[valid]
        y = self._meta_labels(bars, labels)[valid]
        weight = weights.reindex(labels.index)[valid]
        valid_labels = labels[valid]
        intervals = pd.Series(valid_labels["t1"].to_numpy(), index=valid_labels.index)

        x_all = events.to_numpy(dtype=float)
        y_all = y.to_numpy()
        w_all = weight.to_numpy(dtype=float)
        if np.unique(y_all).size < 2:
            raise ValueError("meta-labels are single-class; cannot train a classifier")

        per_fold_rows: list[dict[str, Any]] = []
        pooled: dict[str, dict[str, list[np.ndarray]]] = {
            name: {"y": [], "p": []} for name in _MODELS
        }
        fold_id = 0
        for train_idx, test_idx in splitter.split(events, intervals):
            if train_idx.size == 0 or test_idx.size == 0:
                continue
            y_train = y_all[train_idx]
            if np.unique(y_train).size < 2:
                continue
            for name in _MODELS:
                estimator = self._make_estimator(name)
                self._fit_estimator(estimator, name, x_all[train_idx], y_train, w_all[train_idx])
                prob = self._prob_positive(estimator, x_all[test_idx])
                y_test = y_all[test_idx]
                pooled[name]["y"].append(y_test)
                pooled[name]["p"].append(prob)
                per_fold_rows.append(
                    {
                        "fold": fold_id,
                        "model": name,
                        "n_train": int(train_idx.size),
                        "n_test": int(test_idx.size),
                        "accuracy": _accuracy(y_test, prob),
                        "auc": _auc(y_test, prob),
                    }
                )
            fold_id += 1

        oos_metrics: dict[str, dict[str, float]] = {
            name: _pooled_metrics(pooled[name]["y"], pooled[name]["p"]) for name in _MODELS
        }
        per_fold = pd.DataFrame(per_fold_rows)

        # Refit the *selected* model on all events for prediction + persistence.
        final = self._make_estimator(self.model)
        self._fit_estimator(final, self.model, x_all, y_all, w_all)
        self._fitted = final

        resolved_run_id = run_id if run_id is not None else uuid.uuid4().hex
        artifact_root = (
            Path(artifact_dir)
            if artifact_dir is not None
            else Path(tempfile.mkdtemp(prefix="fe_meta_"))
        )
        artifact_root.mkdir(parents=True, exist_ok=True)
        artifact_path = artifact_root / f"meta_{self.model}_{resolved_run_id}.joblib"
        joblib.dump(final, artifact_path)

        self._log_trial(
            logger,
            splitter,
            oos_metrics,
            n_events=x_all.shape[0],
            n_folds=fold_id,
            run_id=resolved_run_id,
            snapshot_hashes=snapshot_hashes,
            git_sha=git_sha,
            ts=ts,
        )
        return FitResult(oos_metrics=oos_metrics, per_fold=per_fold, model_artifact=artifact_path)

    def predict(self, bars: Bars) -> pd.Series:
        """Return ``p(trade) in [0, 1]`` per bar from the fitted selected model."""
        if self._fitted is None:
            raise RuntimeError("call fit() before predict()")
        features = self._features(bars).reindex(columns=list(_FEATURE_COLUMNS)).fillna(0.0)
        prob = self._prob_positive(self._fitted, features.to_numpy(dtype=float))
        return pd.Series(prob, index=bars.index, name="p_trade")

    # --- provenance ----------------------------------------------------------

    def _config_hash(self) -> str:
        blob = b"FE-METACFG-v1\n" + json.dumps(
            {"model": self.model, "params": self.params, "seed": self.seed},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(blob).hexdigest()

    def _log_trial(
        self,
        logger: TrialLogger,
        splitter: Splitter,
        oos_metrics: dict[str, dict[str, float]],
        *,
        n_events: int,
        n_folds: int,
        run_id: str,
        snapshot_hashes: list[str],
        git_sha: str | None,
        ts: datetime | None,
    ) -> None:
        baseline = "logistic_l2" if self.model == "lightgbm" else "lightgbm"
        metrics = {
            "oos_auc": oos_metrics[self.model]["auc"],
            "oos_accuracy": oos_metrics[self.model]["accuracy"],
            "oos_log_loss": oos_metrics[self.model]["log_loss"],
            "oos_auc_baseline": oos_metrics[baseline]["auc"],
            "n_events": float(n_events),
            "n_folds": float(n_folds),
        }
        logger.log(
            TrialRecord(
                trial_id=f"{run_id}-meta",
                run_id=run_id,
                ts=ts if ts is not None else datetime.now(UTC),
                strategy_family=f"meta:{self.model}:{self.primary.family}",
                config_hash=self._config_hash(),
                params=self.params,
                data_snapshot_hashes=snapshot_hashes,
                cv_scheme=_cv_scheme(splitter),
                metrics=metrics,
                seed=self.seed,
                git_sha=git_sha if git_sha is not None else current_git_sha(),
            )
        )
