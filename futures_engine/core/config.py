"""Typed, validated configuration loading (YAML -> pydantic models).

``instruments.yaml`` (T0), ``costs.yaml`` and ``prop_rules.yaml`` (T2) all have real
validating models here. Unknown keys are rejected everywhere (``extra="forbid"``) --
config typos must fail loudly rather than be silently ignored (Global Constraint G15:
no magic constants; every cost value and prop-rule number lives in YAML, not code).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict

from futures_engine.core.types import InstrumentSpec
from futures_engine.costs.model import CostConfig
from futures_engine.prop.rules import PropRuleSet


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML file and require the top-level document to be a mapping."""
    text = Path(path).read_text(encoding="utf-8")
    data = yaml.safe_load(text)
    if not isinstance(data, dict):
        raise ValueError(
            f"expected a mapping at the top level of {path}, got {type(data).__name__}"
        )
    return data


class InstrumentsConfig(BaseModel):
    """Schema for ``configs/instruments.yaml``: a list of instrument specs."""

    model_config = ConfigDict(extra="forbid")

    instruments: list[InstrumentSpec]


def load_instruments(path: str | Path) -> dict[str, InstrumentSpec]:
    """Load instrument specs, keyed by ``symbol_root``. Rejects duplicates."""
    config = InstrumentsConfig.model_validate(load_yaml(path))
    result: dict[str, InstrumentSpec] = {}
    for spec in config.instruments:
        if spec.symbol_root in result:
            raise ValueError(f"duplicate instrument symbol_root: {spec.symbol_root}")
        result[spec.symbol_root] = spec
    return result


class CostsConfig(BaseModel):
    """Schema for ``configs/costs.yaml``: cost profiles keyed by instrument symbol_root."""

    model_config = ConfigDict(extra="forbid")

    costs: dict[str, CostConfig]


def load_costs(path: str | Path) -> dict[str, CostConfig]:
    """Load per-instrument cost profiles, keyed by ``symbol_root``."""
    return CostsConfig.model_validate(load_yaml(path)).costs


class PropRulesConfig(BaseModel):
    """Schema for ``configs/prop_rules.yaml``: rule bodies keyed by preset name.

    The mapping key is the preset name; it is injected as :attr:`PropRuleSet.name`, so
    the body must not repeat ``name`` (``extra="forbid"`` would reject it).
    """

    model_config = ConfigDict(extra="forbid")

    prop_rules: dict[str, dict[str, Any]]


def load_prop_rules(path: str | Path) -> dict[str, PropRuleSet]:
    """Load prop-rule presets, keyed by (and named after) their mapping key."""
    config = PropRulesConfig.model_validate(load_yaml(path))
    return {name: PropRuleSet(name=name, **body) for name, body in config.prop_rules.items()}


class Settings(BaseModel):
    """Top-level validated settings loaded from a ``configs/`` directory.

    Holds the instrument specs (T0), per-instrument cost profiles and prop-rule
    presets (T2). Every value originates in YAML; nothing is hard-coded here.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    instruments: dict[str, InstrumentSpec]
    costs: dict[str, CostConfig]
    prop_rules: dict[str, PropRuleSet]

    @classmethod
    def load(cls, config_dir: str | Path) -> Settings:
        """Load settings from a directory of ``instruments``/``costs``/``prop_rules`` YAML."""
        cfg_dir = Path(config_dir)
        return cls(
            instruments=load_instruments(cfg_dir / "instruments.yaml"),
            costs=load_costs(cfg_dir / "costs.yaml"),
            prop_rules=load_prop_rules(cfg_dir / "prop_rules.yaml"),
        )
