from hyperopt import fmin, tpe, hp
import numpy as np

def run_backtest(params):
    return -np.abs(params['lookback_period'] - 20)  # Simplified metric

space = {'lookback_period': hp.quniform('lookback_period', 5, 60, 1)}

best_params = fmin(run_backtest, space, algo=tpe.suggest, max_evals=100)
