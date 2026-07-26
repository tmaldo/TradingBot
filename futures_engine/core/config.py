"""Typed, validated configuration loading (YAML -> pydantic models).

Only ``instruments.yaml`` has a real validating model in task T0; ``costs.yaml``
and ``prop_rules.yaml`` are documented stubs whose validating models arrive in
T2, so ``Settings`` deliberately does not load them yet. Unknown keys are
rejected everywhere (``extra="forbid"``) -- config typos must fail loudly rather
than be silently ignored (Global Constraint G15: no magic constants).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict

from futures_engine.core.types import InstrumentSpec


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


class Settings(BaseModel):
    """Top-level validated settings loaded from a ``configs/`` directory.

    In T0 this holds only the (real) instrument specs. Cost and prop-rule
    sections are added by T2 when their schemas are defined.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    instruments: dict[str, InstrumentSpec]

    @classmethod
    def load(cls, config_dir: str | Path) -> Settings:
        """Load settings from a directory containing ``instruments.yaml``."""
        cfg_dir = Path(config_dir)
        return cls(instruments=load_instruments(cfg_dir / "instruments.yaml"))
