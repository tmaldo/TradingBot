"""G1 enforcement: yfinance is a DEV-ONLY source, never in a validation path.

Two repo-wide checks:

1. **AST** -- the only module that actually imports the ``yfinance`` SDK
   (``import yfinance`` / ``from yfinance import ...``) is
   ``futures_engine/data/adapters/yfinance_dev.py``.
2. **Substring** -- the string ``yfinance`` does not appear anywhere in the
   research / backtest / validation / feature / label / prop / report / pipeline
   packages (the paths that decide whether a strategy is believed).

Together these pin the constraint that dev-only data can never leak into a
research, backtest, or validation result.
"""

from __future__ import annotations

import ast
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_PKG = _ROOT / "futures_engine"
_ALLOWED_IMPORTER = _PKG / "data" / "adapters" / "yfinance_dev.py"

# Packages on the research/backtest/validation path: none may mention yfinance.
_VALIDATION_PACKAGES = (
    "research",
    "backtest",
    "validation",
    "features",
    "labels",
    "prop",
    "sizing",
    "regime",
    "costs",
    "report",
    "pipeline",
)


def _imports_yfinance(tree: ast.AST) -> bool:
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            if any(
                alias.name == "yfinance" or alias.name.startswith("yfinance.")
                for alias in node.names
            ):
                return True
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            if mod == "yfinance" or mod.startswith("yfinance."):
                return True
    return False


def test_only_yfinance_dev_imports_the_sdk() -> None:
    offenders = []
    for path in _PKG.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        if _imports_yfinance(tree) and path != _ALLOWED_IMPORTER:
            offenders.append(str(path.relative_to(_ROOT)))
    assert offenders == [], f"yfinance SDK imported outside yfinance_dev.py: {offenders}"


def test_yfinance_dev_actually_imports_the_sdk() -> None:
    # Sanity: the allow-listed file really is the SDK importer (guard against a
    # false-pass if the module is ever renamed/removed).
    tree = ast.parse(_ALLOWED_IMPORTER.read_text(encoding="utf-8"))
    assert _imports_yfinance(tree)


def test_validation_paths_never_mention_yfinance() -> None:
    offenders = []
    for pkg in _VALIDATION_PACKAGES:
        for path in (_PKG / pkg).rglob("*.py"):
            if "yfinance" in path.read_text(encoding="utf-8"):
                offenders.append(str(path.relative_to(_ROOT)))
    assert offenders == [], f"yfinance referenced in a validation-path package: {offenders}"
