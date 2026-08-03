"""Thin form -> validated :class:`PipelineConfig` builder for the Configure screen.

The cockpit never lets a user hand-edit YAML: the browser form is parsed here and
handed straight to the EXISTING :class:`~futures_engine.pipeline.run.PipelineConfig`
pydantic model, which is the single source of validation (UI-G1). This module adds
only two things pydantic cannot express from raw form strings:

* a small **grid parser** (``"name: v1, v2, ..."`` lines -> ``dict[str, list[int]]``);
  malformed text raises a clear :class:`ValueError` the route surfaces inline --
  entries are never silently dropped.
* the **honest-DSR** ``strategy_family`` derivation: the family is sourced from the
  chosen signal's own ``.family`` (``SIGNAL_REGISTRY[signal]().family``), never read
  from the form, so a user cannot mismatch it and collapse the DSR trial count (the
  exact failure the T9 fix closed). An unknown signal fails loudly here.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import yaml

from futures_engine.backtest.strategy_adapter import SIGNAL_REGISTRY
from futures_engine.core.config import load_yaml
from futures_engine.pipeline.run import PipelineConfig

# Instruments the cockpit offers (resolve against configs/instruments.yaml at run
# time; MES/MNQ are the only micro contracts the engine ships specs for).
INSTRUMENTS: tuple[str, ...] = ("MES", "MNQ")

# Optional scalar fields we forward verbatim to the pydantic model (which coerces
# the string form values and enforces every bound). Omitted keys fall back to the
# model's own defaults.
_SCALAR_FIELDS: tuple[str, ...] = (
    "seed",
    "n_splits",
    "embargo_frac",
    "pbo_partitions",
    "bootstrap_n",
)


def signal_choices() -> list[str]:
    """Return the selectable signal keys (the :data:`SIGNAL_REGISTRY` keys)."""
    return list(SIGNAL_REGISTRY)


def prop_preset_choices(configs_dir: str | Path) -> list[str]:
    """Return the prop-preset names from ``configs/prop_rules.yaml`` (``prop_rules:``)."""
    data = load_yaml(Path(configs_dir) / "prop_rules.yaml")
    return list(data.get("prop_rules", {}))


def derive_strategy_family(signal: str) -> str:
    """Return the honest ``strategy_family`` for ``signal`` (its ``.family``).

    Raises :class:`ValueError` for a signal not in :data:`SIGNAL_REGISTRY`, so an
    unknown selection is rejected before it can reach the pipeline.
    """
    try:
        signal_cls = SIGNAL_REGISTRY[signal]
    except KeyError:
        known = ", ".join(SIGNAL_REGISTRY)
        raise ValueError(f"unknown signal {signal!r}; choose one of: {known}") from None
    return str(signal_cls().family)


def parse_grid(text: str) -> dict[str, list[int]]:
    """Parse a grid textarea into ``dict[str, list[int]]``.

    Format: one ``name: v1, v2, v3`` line per parameter axis. Malformed input
    (empty, missing ``:``, empty name, no values, or a non-integer value) raises a
    clear :class:`ValueError` -- bad entries are never silently dropped.
    """
    lines = [line.strip() for line in text.strip().splitlines() if line.strip()]
    if not lines:
        raise ValueError("grid is empty: add at least one 'name: v1, v2, ...' line")
    grid: dict[str, list[int]] = {}
    for line in lines:
        if ":" not in line:
            raise ValueError(f"malformed grid line (missing ':'): {line!r}")
        key, _, rest = line.partition(":")
        key = key.strip()
        if not key:
            raise ValueError(f"malformed grid line (empty parameter name): {line!r}")
        if key in grid:
            raise ValueError(f"duplicate grid parameter {key!r}")
        raw = [tok.strip() for tok in rest.split(",") if tok.strip()]
        if not raw:
            raise ValueError(f"grid parameter {key!r} has no values")
        try:
            grid[key] = [int(tok) for tok in raw]
        except ValueError:
            raise ValueError(
                f"grid parameter {key!r} has a non-integer value in {rest.strip()!r}"
            ) from None
    return grid


def form_to_config(form: Mapping[str, str]) -> PipelineConfig:
    """Parse the Configure form and validate it via :class:`PipelineConfig` (UI-G1).

    Grid text is parsed here; ``strategy_family`` is derived from the signal (never
    trusted from the form). Everything else is forwarded as-is to the pydantic model,
    which coerces the string values and enforces every constraint. Any bad input
    raises :class:`ValueError` (grid/unknown-signal) or ``pydantic.ValidationError``,
    both caught and rendered inline by the route -- never a silent coercion.
    """
    signal = form.get("signal", "")
    # Honest-DSR: derive the family from the signal itself; a form-supplied
    # ``strategy_family`` (if any) is deliberately ignored here.
    strategy_family = derive_strategy_family(signal)

    payload: dict[str, object] = {
        "run_id": form.get("run_id", ""),
        "instrument": form.get("instrument", ""),
        "signal": signal,
        "strategy_family": strategy_family,
        "grid": parse_grid(form.get("grid", "")),
        "snapshot_hash": form.get("snapshot_hash", ""),
        "prop_preset": form.get("prop_preset", ""),
        "survival": {
            "n_paths": form.get("survival_n_paths", ""),
            "horizon_days": form.get("survival_horizon_days", ""),
            "contracts": form.get("survival_contracts", ""),
        },
    }
    for field in _SCALAR_FIELDS:
        if field in form:
            payload[field] = form[field]

    gates = {
        model_key: form[form_key]
        for model_key, form_key in (
            ("min_dsr_p", "gate_min_dsr_p"),
            ("max_pbo", "gate_max_pbo"),
            ("min_survival", "gate_min_survival"),
        )
        if form_key in form
    }
    if gates:
        payload["gates"] = gates

    return PipelineConfig.model_validate(payload)


def config_to_yaml(cfg: PipelineConfig, path: str | Path) -> None:
    """Write ``cfg`` to ``path`` as YAML (round-trips through ``PipelineConfig.load``).

    Creates parent directories as needed. The file is the U5 handoff artifact.
    """
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(
        yaml.safe_dump(cfg.model_dump(mode="python"), sort_keys=False),
        encoding="utf-8",
    )
