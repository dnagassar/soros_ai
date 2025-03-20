# modules/optimization.py
import optuna
import pandas as pd
from autogluon.tabular import TabularPredictor
import mlflow

def objective(trial):
    hyperparameters = {
        'GBM': {
            'num_boost_round': trial.suggest_int('num_boost_round', 100, 500),
            'learning_rate': trial.suggest_loguniform('learning_rate', 1e-3, 1e-1),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
        }
    }
    df = pd.read_csv('../data/historical_prices.csv')
    df['target'] = df['Close'].shift(-1)
    df.dropna(inplace=True)
    train_data = df.iloc[:-50]
    val_data = df.iloc[-50:]
    mlflow.start_run()
    predictor = TabularPredictor(label='target', eval_metric='rmse').fit(
        train_data=train_data,
        hyperparameters=hyperparameters,
        presets='medium_quality',
        verbosity=3,
        time_limit=600,
        raise_on_no_models_fitted=False
    )
    performance = predictor.evaluate(val_data)
    rmse = performance.get('rmse', float('inf'))
    mlflow.log_params(hyperparameters['GBM'])
    mlflow.log_metric("rmse", rmse)
    mlflow.end_run()
    return rmse

def optimize_autogluon(n_trials=20):
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)
    print("Best Hyperparameters:", study.best_params)
    print("Best RMSE:", study.best_value)
    return study.best_params, study.best_value

if __name__ == "__main__":
    best_params, best_rmse = optimize_autogluon(n_trials=20)
