"""Typed, validated live-execution configuration (``configs/live.yaml`` -> pydantic).

Every threshold the live risk layer, kill switches and shutdown monitor use is
loaded and validated here (Global Constraint G15: no magic constants in code).
Unknown keys are rejected (``extra="forbid"``) so a config typo fails loudly.

The kill-switch buffers/margins/rates live in this file; the firm's daily-loss
and trailing-drawdown *amounts* come from ``configs/prop_rules.yaml`` via the
referenced :attr:`AccountConfig.prop_preset`, keeping one source of truth for the
prop-firm rules.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from futures_engine.core.config import load_yaml
from futures_engine.sizing.position import EdgeStats, SizingConfig


class DailyLossLimitConfig(BaseModel):
    """Buffer (USD) held back from the firm's hard daily-loss limit."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    buffer_usd: float = Field(ge=0.0)


class TrailingDdGuardConfig(BaseModel):
    """Margin (USD) held above the trailing-drawdown floor before rejecting."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    margin_usd: float = Field(ge=0.0)


class StaleDataHaltConfig(BaseModel):
    """Maximum age (seconds) of the last tick before data is deemed stale."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    max_age_s: float = Field(gt=0.0)


class FlattenOnDisconnectConfig(BaseModel):
    """Whether a websocket drop raises an alarm alongside the queued flatten."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    alarm: bool = True


class MaxOrderRateConfig(BaseModel):
    """Maximum number of new orders permitted in a rolling 60-second window."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    max_per_minute: int = Field(gt=0)


class KillSwitchConfig(BaseModel):
    """The five kill switches' thresholds (each independently configured)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    daily_loss_limit: DailyLossLimitConfig
    trailing_dd_guard: TrailingDdGuardConfig
    stale_data_halt: StaleDataHaltConfig
    flatten_on_disconnect: FlattenOnDisconnectConfig
    max_order_rate: MaxOrderRateConfig


class EdgeConfig(BaseModel):
    """Bernoulli edge for the sizer, mirroring :class:`EdgeStats`."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    p_win: float = Field(ge=0.0, le=1.0)
    avg_win: float = Field(gt=0.0)
    avg_loss: float = Field(gt=0.0)

    def to_edge_stats(self) -> EdgeStats:
        """Build the T7 :class:`EdgeStats` value object."""
        return EdgeStats(p_win=self.p_win, avg_win=self.avg_win, avg_loss=self.avg_loss)


class LiveSizingConfig(BaseModel):
    """Inputs to the T7 sizer supplied to the RiskManager in the live flow.

    The RiskManager rejects any order whose qty exceeds
    :func:`~futures_engine.sizing.position.position_size` evaluated on these.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    vol_estimate_usd: float = Field(gt=0.0)
    target_daily_vol_usd: float = Field(gt=0.0)
    kelly_fraction_cap: float = Field(gt=0.0, le=0.25)
    max_contracts: int = Field(ge=0)
    survival_max_contracts: int = Field(ge=0)
    edge: EdgeConfig

    def to_sizing_config(self) -> SizingConfig:
        """Build the T7 :class:`SizingConfig` value object."""
        return SizingConfig(
            target_daily_vol_usd=self.target_daily_vol_usd,
            kelly_fraction_cap=self.kelly_fraction_cap,
            max_contracts=self.max_contracts,
        )


class RiskConfig(BaseModel):
    """Kill-switch thresholds and sizing inputs for the RiskManager."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    kill_switches: KillSwitchConfig
    sizing: LiveSizingConfig


class ShutdownConfig(BaseModel):
    """Shutdown criteria for the live monitor (read from config, never code)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    max_slippage_divergence_usd: float = Field(gt=0.0)
    rolling_sharpe_floor: float
    rolling_sharpe_min_samples: int = Field(gt=0)
    rolling_window: int = Field(gt=0)
    drift_threshold_z: float = Field(gt=0.0)


class AccountConfig(BaseModel):
    """Account identity: starting equity and the prop-rule preset to guard against."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    starting_balance: float = Field(gt=0.0)
    prop_preset: str = Field(min_length=1)


class TradovateAdapterConfig(BaseModel):
    """Tradovate demo/paper endpoints and account identity (MFFU route)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    base_url: str = Field(min_length=1)
    ws_url: str = Field(min_length=1)
    account_spec: str = Field(min_length=1)
    account_id: int = Field(gt=0)


class TopstepXAdapterConfig(BaseModel):
    """TopstepX / ProjectX gateway demo endpoint and account identity."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    base_url: str = Field(min_length=1)
    account_id: int = Field(gt=0)


class AdaptersConfig(BaseModel):
    """Per-venue adapter configuration."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    tradovate: TradovateAdapterConfig
    topstepx: TopstepXAdapterConfig


class LiveConfig(BaseModel):
    """Top-level validated live configuration loaded from ``configs/live.yaml``."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    account: AccountConfig
    risk: RiskConfig
    shutdown: ShutdownConfig
    adapters: AdaptersConfig

    @classmethod
    def load(cls, path: str | Path) -> LiveConfig:
        """Load and validate ``configs/live.yaml`` (unknown keys rejected)."""
        return cls.model_validate(load_yaml(path))
