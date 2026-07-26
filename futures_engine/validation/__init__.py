"""Validation & statistics: leak-proof CV splitters and anti-overfitting stats.

This package implements Global Constraints G9 (purged k-fold + embargo,
walk-forward, CPCV; plain k-fold forbidden) and G10 (Deflated Sharpe Ratio,
Probability of Backtest Overfitting, bootstrap Sharpe CIs, red-flag reporter).
Pure numpy/pandas/arch logic with no data-layer dependencies.
"""
