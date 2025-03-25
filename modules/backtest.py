# Create a new file at modules/backtest.py with this content:
# This file bridges the actual implementation in backtests/backtest.py

from backtests.backtest import (
    BacktestEngine,
    run_full_backtest,
    parameter_sweep,
    evaluate_backtest
)

# Re-export the needed functionality
__all__ = ['BacktestEngine', 'run_full_backtest', 'parameter_sweep', 'evaluate_backtest']