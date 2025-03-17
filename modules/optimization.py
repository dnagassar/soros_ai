# optimization.py
import optuna
import pandas as pd
from autogluon.tabular import TabularPredictor

def objective(trial):
    hyperparameters = {
        'GBM': {
            'num_boost_round': trial.suggest_int('num_boost_round', 100, 500),
            'learning_rate': trial.suggest_loguniform('learning_rate', 1e-3, 1e-1),
        },
        'NN': {
            'num_epochs': trial.suggest_int('num_epochs', 5, 50),
            'learning_rate': trial.suggest_loguniform('nn_lr', 1e-4, 1e-2),
        }
    }

    df = pd.read_csv('../data/historical_prices.csv')
    df['target'] = df['Close'].shift(-1)
    df.dropna(inplace=True)

    train_data = df.iloc[:-50]
    val_data = df.iloc[-50:]

    predictor = TabularPredictor(label='target', eval_metric='rmse').fit(
        train_data,
        hyperparameters=hyperparameters,
        presets='medium_quality',
        verbosity=0
    )

    performance = predictor.evaluate(val_data)
    rmse = performance['rmse']
    return performance['rmse']

def optimize_autogluon(n_trials=20):
    import optuna
    study = optuna.create_study(direction='minimize')
    study = study = optuna.create_study()
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=20)

    print("Best Hyperparameters:", study.best_params)
    print("Best RMSE:", study.best_value)

if __name__ == "__main__":
    optimize_autogluon_params()
