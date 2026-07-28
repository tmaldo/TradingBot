"""The verdict is the crux (G16): unit-test GO/NO-GO on BOTH sides of EACH gate.

Every test holds all other gates at a comfortable pass and flips exactly one
gate across its boundary (just-pass vs just-fail), asserting the overall verdict
flips with it. The default :class:`GateConfig` thresholds are 0.95 / 0.5 / 0.90.
"""

from __future__ import annotations

from futures_engine.report.builder import GateConfig, decide_verdict
from futures_engine.validation.stats import RedFlag

# A comfortable all-pass baseline; individual tests perturb one axis at a time.
_PASS_DSR = 0.99
_PASS_PBO = 0.10
_PASS_SURVIVAL = 0.97
_NO_FLAGS: list[RedFlag] = []
_GATES = GateConfig()  # defaults 0.95 / 0.5 / 0.90


def _verdict(
    *,
    dsr_p: float = _PASS_DSR,
    pbo: float = _PASS_PBO,
    p_survival: float = _PASS_SURVIVAL,
    red_flags: list[RedFlag] | None = None,
) -> str:
    return decide_verdict(
        dsr_p=dsr_p,
        pbo=pbo,
        p_survival=p_survival,
        red_flags=red_flags if red_flags is not None else _NO_FLAGS,
        gates=_GATES,
    ).decision


def test_all_pass_is_go() -> None:
    assert _verdict() == "GO"


# --- survival gate boundary (>= 0.90) ---------------------------------------


def test_survival_just_passes() -> None:
    assert _verdict(p_survival=0.90) == "GO"


def test_survival_just_fails() -> None:
    assert _verdict(p_survival=0.8999) == "NO-GO"


# --- DSR gate boundary (>= 0.95) --------------------------------------------


def test_dsr_just_passes() -> None:
    assert _verdict(dsr_p=0.95) == "GO"


def test_dsr_just_fails() -> None:
    assert _verdict(dsr_p=0.9499) == "NO-GO"


# --- PBO gate boundary (<= 0.5) ---------------------------------------------


def test_pbo_just_passes() -> None:
    assert _verdict(pbo=0.50) == "GO"


def test_pbo_just_fails() -> None:
    assert _verdict(pbo=0.5001) == "NO-GO"


# --- fail-severity red-flag gate --------------------------------------------


def test_warn_flag_alone_stays_go() -> None:
    warn = RedFlag(code="SHARPE_IMPLAUSIBLE", message="high but only a warning", severity="warn")
    assert _verdict(red_flags=[warn]) == "GO"


def test_fail_flag_forces_no_go() -> None:
    fail = RedFlag(code="EDGE_FAILS_DELAY", message="edge dies under delay", severity="fail")
    assert _verdict(red_flags=[fail]) == "NO-GO"


def test_fail_flag_forces_no_go_even_when_all_stats_pass() -> None:
    fail = RedFlag(code="EDGE_FAILS_COSTS", message="edge eaten by costs", severity="fail")
    v = decide_verdict(
        dsr_p=0.999,
        pbo=0.0,
        p_survival=1.0,
        red_flags=[fail],
        gates=_GATES,
    )
    assert v.decision == "NO-GO"
    assert v.is_go is False
    assert "EDGE_FAILS_COSTS" in v.fail_flag_codes


# --- custom gate config -----------------------------------------------------


def test_custom_gate_config_thresholds_apply() -> None:
    strict = GateConfig(min_dsr_p=0.80, max_pbo=0.30, min_survival=0.50)
    # survival 0.60 passes the relaxed 0.50 bar but would fail the default 0.90.
    assert (
        decide_verdict(dsr_p=0.85, pbo=0.25, p_survival=0.60, red_flags=[], gates=strict).decision
        == "GO"
    )
    assert (
        decide_verdict(dsr_p=0.85, pbo=0.25, p_survival=0.60, red_flags=[], gates=_GATES).decision
        == "NO-GO"
    )


def test_gate_config_defaults() -> None:
    g = GateConfig()
    assert g.min_dsr_p == 0.95
    assert g.max_pbo == 0.5
    assert g.min_survival == 0.90
