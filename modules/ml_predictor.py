# modules/ml_predictor.py
import pandas as pd
import numpy as np
from autogluon.tabular import TabularPredictor

def flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    If the DataFrame has MultiIndex columns (e.g. from yfinance),
    flatten them to a single level by choosing the first non-empty level.
    Also convert all column names to strings and strip whitespace.
    """
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] if col[0] != '' else col[1] for col in df.columns]
    df.columns = [str(col).strip() for col in df.columns]
    return df

def train_ensemble_model(train_data: pd.DataFrame, label_column='target', time_limit=1200, hyperparameters=None):
    """
    Trains an AutoGluon ensemble model on the provided training data.
    Ensures that the DataFrame is flattened and that the label column exists.
    
    Parameters:
      - train_data (pd.DataFrame): Training data with features and the target.
      - label_column (str): Name of the target variable.
      - time_limit (int): Training time limit in seconds.
      - hyperparameters (dict): Optional hyperparameters.
      
    Returns:
      - predictor: The trained TabularPredictor model.
    """
    if hyperparameters is None:
        hyperparameters = {}
    # Flatten the DataFrame to ensure column names are single-level.
    train_data = flatten_columns(train_data)
    print("Starting training with data shape:", train_data.shape)
    print("Training data columns:", train_data.columns.tolist())
    
    if label_column not in train_data.columns:
        raise ValueError(f"Training data must contain the label column: {label_column}")
    
    predictor = TabularPredictor(label=label_column, eval_metric='rmse').fit(
        train_data=train_data,
        presets='best_quality',
        time_limit=time_limit,
        hyperparameters=hyperparameters,
        verbosity=3,
        raise_on_no_models_fitted=False
    )
    
    models_info = predictor.info().get('models')
    if not models_info:
        print("Warning: No models were trained successfully during fit().")
    else:
        print("Trained models:", list(models_info.keys()))
    return predictor

def predict_with_model(predictor, test_data: pd.DataFrame):
    """
    Uses the trained predictor to make predictions on test_data.
    If no models were trained, returns a dummy Series of zeros.
    
    Parameters:
      - predictor: The trained TabularPredictor model.
      - test_data (pd.DataFrame): Data for generating predictions.
      
    Returns:
      - predictions: A pandas Series of predictions.
    """
    test_data = flatten_columns(test_data)
    if not predictor.info().get('models'):
        print("No models available; returning zeros as dummy predictions.")
        return pd.Series(np.zeros(len(test_data)))
    predictions = predictor.predict(test_data)
    return predictions

def ensemble_predict(X_train, y_train, X_test, time_limit=1200, hyperparameters=None):
    """
    Trains an ensemble model using X_train and y_train, then returns predictions on X_test.
    Ensures proper alignment and flattening of columns.
    If no models are trained, returns a dummy Series of zeros.
    
    Parameters:
      - X_train: Training features (array-like or DataFrame).
      - y_train: Training targets (array-like or Series).
      - X_test: Test features (array-like or DataFrame).
      - time_limit: Maximum training time in seconds.
      - hyperparameters: Optional hyperparameters.
      
    Returns:
      - predictions: Predictions as a pandas Series.
    """
    if not isinstance(X_train, pd.DataFrame):
        X_train = pd.DataFrame(X_train)
    if not isinstance(y_train, pd.Series):
        y_train = pd.Series(y_train)
    X_train = flatten_columns(X_train.copy())
    y_train = y_train.reindex(X_train.index)
    X_train['target'] = y_train.astype(float)
    X_train = flatten_columns(X_train)
    
    predictor = train_ensemble_model(X_train, time_limit=time_limit, hyperparameters=hyperparameters)
    if not predictor.info().get('models'):
        return pd.Series(np.zeros(len(X_test)))
    if not isinstance(X_test, pd.DataFrame):
        X_test = pd.DataFrame(X_test)
    X_test = flatten_columns(X_test)
    predictions = predict_with_model(predictor, X_test)
    return predictions

if __name__ == "__main__":
    # Example usage for debugging:
    df = pd.read_csv('../data/historical_prices.csv')
    df = flatten_columns(df)
    df['target'] = df['Close'].shift(-1)
    df.dropna(inplace=True)
    train_data = df.iloc[:-20]
    test_data = df.iloc[-20:].drop(columns=['target'])
    predictor = train_ensemble_model(train_data=train_data, time_limit=1200)
    preds = predict_with_model(predictor, test_data)
    print("Ensemble Predictions:", preds)
