# modules/optimization.py
"""
Optimization module for parameter optimization using various methods
"""
import numpy as np
import pandas as pd
import itertools
import logging
import os
import json
import time
import datetime
import concurrent.futures
from functools import partial
import optuna
from optuna.samplers import TPESampler
from autogluon.tabular import TabularPredictor
from config import (
    SystemConfig, 
    GlobalSettings, 
    get_strategy_parameters,
    get_risk_parameters
)
from modules.data_acquisition import fetch_price_data, batch_fetch_price_data
from modules.logger import setup_logger
from modules.risk_manager import RiskManager
from modules.backtest import run_full_backtest

# Configure logging
logger = setup_logger(__name__)

# Result directory
RESULTS_DIR = os.path.join(SystemConfig.DEFAULT_RESULTS_DIR, 'optimization')
os.makedirs(RESULTS_DIR, exist_ok=True)

def evaluate_backtest(params, symbols, start_date, end_date, benchmark_symbol='^GSPC', initial_cash=100000):
    """
    Run a backtest with the given parameters and return performance metrics
    
    Parameters:
      - params: Strategy parameters
      - symbols: List of symbols to trade
      - start_date: Start date for backtest
      - end_date: End date for backtest
      - benchmark_symbol: Benchmark symbol
      - initial_cash: Initial cash for backtest
      
    Returns:
      - dict: Performance metrics
    """
    try:
        # Create risk manager with parameters
        risk_params = {
            'max_position_size': params.get('max_position_size', 0.2),
            'max_portfolio_risk': params.get('max_portfolio_risk', 0.02),
            'stop_loss_pct': params.get('stop_loss', 0.03),
            'take_profit_pct': params.get('take_profit', 0.05),
            'position_sizing_method': params.get('position_sizing_method', 'volatility')
        }
        
        risk_manager = RiskManager(**risk_params)
        
        # Run backtest
        metrics, benchmark, _ = run_full_backtest(
            start_date=start_date,
            end_date=end_date,
            assets=symbols,
            strategy_params=params,
            benchmark_symbol=benchmark_symbol,
            initial_cash=initial_cash,
            risk_manager=risk_manager
        )
        
        # Return important metrics
        return {
            'sharpe': metrics.get('sharpe', 0),
            'pnl_percent': metrics.get('pnl_percent', 0),
            'max_drawdown': metrics.get('max_drawdown', 0),
            'win_rate': metrics.get('win_rate', 0),
            'sqn': metrics.get('sqn', 0),
            'alpha': metrics.get('pnl_percent', 0) - benchmark.get('return_percent', 0)
        }
    
    except Exception as e:
        logger.error(f"Error in backtest evaluation: {e}")
        return {
            'sharpe': 0,
            'pnl_percent': 0,
            'max_drawdown': 100,  # Penalize errors
            'win_rate': 0,
            'sqn': 0,
            'alpha': -100  # Penalize errors
        }

def objective(trial, symbols, start_date, end_date, param_space, objective_metric='sharpe'):
    """
    Objective function for Optuna optimization
    
    Parameters:
      - trial: Optuna trial object
      - symbols: List of symbols to trade
      - start_date: Start date for backtest
      - end_date: End date for backtest
      - param_space: Parameter space for optimization
      - objective_metric: Metric to optimize
      
    Returns:
      - float: Objective value to maximize
    """
    # Generate parameters for this trial
    params = {}
    
    for param_name, param_config in param_space.items():
        param_type = param_config.get('type', 'float')
        
        if param_type == 'float':
            params[param_name] = trial.suggest_float(
                param_name, 
                param_config.get('low'), 
                param_config.get('high'),
                log=param_config.get('log', False)
            )
        
        elif param_type == 'int':
            params[param_name] = trial.suggest_int(
                param_name,
                param_config.get('low'),
                param_config.get('high'),
                step=param_config.get('step', 1),
                log=param_config.get('log', False)
            )
        
        elif param_type == 'categorical':
            params[param_name] = trial.suggest_categorical(
                param_name,
                param_config.get('choices', [])
            )
    
    # Run backtest with these parameters
    metrics = evaluate_backtest(params, symbols, start_date, end_date)
    
    # Save trial results
    for metric_name, value in metrics.items():
        trial.set_user_attr(metric_name, value)
    
    # Return the objective metric
    objective_value = metrics.get(objective_metric, 0)
    
    # Invert sign for metrics where lower is better
    if objective_metric in ['max_drawdown']:
        objective_value = -objective_value
    
    return objective_value

def optimize_parameters(symbols, start_date, end_date, param_space, n_trials=50, objective_metric='sharpe'):
    """
    Optimize strategy parameters using Optuna
    
    Parameters:
      - symbols: List of symbols to trade
      - start_date: Start date for backtest
      - end_date: End date for backtest
      - param_space: Parameter space for optimization
      - n_trials: Number of optimization trials
      - objective_metric: Metric to optimize
      
    Returns:
      - dict: Optimized parameters and results
    """
    logger.info(f"Starting parameter optimization with {n_trials} trials")
    logger.info(f"Optimizing for {objective_metric}")
    
    # Create study
    study_name = f"strategy_optimization_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Set seed for reproducibility
    seed = 42
    
    # Create study
    study = optuna.create_study(
        study_name=study_name,
        direction="maximize",
        sampler=TPESampler(seed=seed)
    )
    
    # Run optimization
    objective_func = partial(
        objective,
        symbols=symbols,
        start_date=start_date,
        end_date=end_date,
        param_space=param_space,
        objective_metric=objective_metric
    )
    
    study.optimize(objective_func, n_trials=n_trials)
    
    # Get best parameters
    best_params = study.best_params
    best_value = study.best_value
    best_trial = study.best_trial
    
    # Get all metrics from best trial
    best_metrics = {
        'sharpe': best_trial.user_attrs.get('sharpe', 0),
        'pnl_percent': best_trial.user_attrs.get('pnl_percent', 0),
        'max_drawdown': best_trial.user_attrs.get('max_drawdown', 0),
        'win_rate': best_trial.user_attrs.get('win_rate', 0),
        'sqn': best_trial.user_attrs.get('sqn', 0),
        'alpha': best_trial.user_attrs.get('alpha', 0)
    }
    
    # Get trial history for visualization
    trials_df = study.trials_dataframe()
    
    # Save results
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    results_file = os.path.join(RESULTS_DIR, f"optim_results_{timestamp}.json")
    
    results = {
        'study_name': study_name,
        'objective_metric': objective_metric,
        'n_trials': n_trials,
        'best_params': best_params,
        'best_value': best_value,
        'best_metrics': best_metrics,
        'timestamp': timestamp,
        'symbols': symbols,
        'start_date': start_date,
        'end_date': end_date,
        'param_space': param_space
    }
    
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=4, default=str)
    
    # Save trials data
    trials_file = os.path.join(RESULTS_DIR, f"optim_trials_{timestamp}.csv")
    trials_df.to_csv(trials_file, index=False)
    
    logger.info(f"Optimization complete. Best value: {best_value:.4f}")
    logger.info(f"Best parameters: {best_params}")
    logger.info(f"Results saved to {results_file}")
    
    return results

def grid_search(param_grid, symbols, start_date, end_date, n_jobs=None, objective_metric='sharpe'):
    """
    Perform grid search for parameter optimization
    
    Parameters:
      - param_grid: Parameter grid as a dictionary of parameter names to lists of values
      - symbols: List of symbols to trade
      - start_date: Start date for backtest
      - end_date: End date for backtest
      - n_jobs: Number of parallel jobs (None for all available)
      - objective_metric: Metric to optimize
      
    Returns:
      - pd.DataFrame: Results of grid search
    """
    logger.info(f"Starting grid search with parameter grid: {param_grid}")
    
    # Generate all combinations of parameters
    param_names = list(param_grid.keys())
    param_values = list(itertools.product(*[param_grid[name] for name in param_names]))
    params_list = [dict(zip(param_names, values)) for values in param_values]
    
    logger.info(f"Generated {len(params_list)} parameter combinations")
    
    # Evaluate each combination
    results = []
    
    if n_jobs is None or n_jobs > 1:
        # Parallel execution
        with concurrent.futures.ProcessPoolExecutor(max_workers=n_jobs) as executor:
            evaluate_func = partial(evaluate_backtest, 
                                  symbols=symbols, 
                                  start_date=start_date, 
                                  end_date=end_date)
            
            for params, metrics in zip(params_list, executor.map(evaluate_func, params_list)):
                result = {**params, **metrics}
                results.append(result)
                logger.debug(f"Completed evaluation with params: {params}")
    else:
        # Sequential execution
        for params in params_list:
            metrics = evaluate_backtest(params, symbols, start_date, end_date)
            result = {**params, **metrics}
            results.append(result)
            logger.debug(f"Completed evaluation with params: {params}")
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    # Sort by objective metric
    if objective_metric in ['max_drawdown']:
        # Lower is better
        results_df = results_df.sort_values(objective_metric)
    else:
        # Higher is better
        results_df = results_df.sort_values(objective_metric, ascending=False)
    
    # Save results
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    results_file = os.path.join(RESULTS_DIR, f"grid_search_results_{timestamp}.csv")
    results_df.to_csv(results_file, index=False)
    
    logger.info(f"Grid search complete. Results saved to {results_file}")
    
    return results_df

def optimize_autogluon(param_grid, n_trials=10, train_frac=0.8, time_limit=600, 
                     metric='sharpe', presets='medium_quality'):
    """
    Optimize parameters using AutoGluon (meta-learning approach)
    
    Parameters:
      - param_grid: Parameter grid as a dictionary of parameter names to lists of values
      - n_trials: Number of random trials to run for initial data collection
      - train_frac: Fraction of data to use for training
      - time_limit: Time limit for AutoGluon in seconds
      - metric: Metric to optimize
      - presets: AutoGluon presets
      
    Returns:
      - dict: Optimized parameters
    """
    logger.info(f"Starting AutoGluon optimization with {n_trials} initial trials")
    
    # Generate random parameter combinations for initial data collection
    param_names = list(param_grid.keys())
    
    training_data = []
    
    for _ in range(n_trials):
        # Generate random parameters
        params = {}
        for param_name in param_names:
            values = param_grid[param_name]
            params[param_name] = np.random.choice(values)
        
        # Run backtest with these parameters
        symbols = ['AAPL', 'MSFT', 'AMZN', 'GOOGL', 'META']  # Example symbols
        start_date = '2022-01-01'
        end_date = '2023-01-01'
        
        metrics = evaluate_backtest(params, symbols, start_date, end_date)
        
        # Add to training data
        data_point = {**params, **metrics}
        training_data.append(data_point)
        
        logger.debug(f"Completed evaluation with params: {params}")
    
    # Convert to DataFrame
    train_df = pd.DataFrame(training_data)
    
    # Split into features and target
    X = train_df[param_names]
    y = train_df[metric]
    
    # Adjust target for metrics where lower is better
    if metric in ['max_drawdown']:
        y = -y
    
    # Split into train and validation
    n_train = int(len(X) * train_frac)
    X_train, X_val = X.iloc[:n_train], X.iloc[n_train:]
    y_train, y_val = y.iloc[:n_train], y.iloc[n_train:]
    
    # Train AutoGluon model
    logger.info("Training AutoGluon model")
    
    # Create temporary directory for AutoGluon
    import tempfile
    with tempfile.TemporaryDirectory() as temp_dir:
        predictor = TabularPredictor(
            label=metric,
            path=temp_dir,
            problem_type='regression'
        )
        
        # Train model
        predictor.fit(
            train_data=pd.concat([X_train, y_train], axis=1),
            presets=presets,
            time_limit=time_limit
        )
        
        # Generate all parameter combinations
        param_values = list(itertools.product(*[param_grid[name] for name in param_names]))
        all_params = [dict(zip(param_names, values)) for values in param_values]
        
        # Create DataFrame for prediction
        all_params_df = pd.DataFrame(all_params)
        
        # Predict performance for all combinations
        predictions = predictor.predict(all_params_df)
        
        # Find best parameters
        if metric in ['max_drawdown']:
            # Lower is better (but we inverted the sign)
            best_idx = predictions.argmax()
        else:
            # Higher is better
            best_idx = predictions.argmax()
        
        best_params = all_params[best_idx]
        predicted_value = predictions.iloc[best_idx]
        
        logger.info(f"Optimization complete. Predicted {metric}: {predicted_value:.4f}")
        logger.info(f"Best parameters: {best_params}")
    
    # Save results
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    results_file = os.path.join(RESULTS_DIR, f"autogluon_results_{timestamp}.json")
    
    results = {
        'best_params': best_params,
        'predicted_value': float(predicted_value),
        'metric': metric,
        'n_trials': n_trials,
        'timestamp': timestamp,
        'param_grid': param_grid
    }
    
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=4, default=str)
    
    logger.info(f"Results saved to {results_file}")
    
    return best_params

def run_optimization(optimization_type='optuna', param_config=None):
    """
    Run parameter optimization with the specified method
    
    Parameters:
      - optimization_type: Type of optimization to run ('optuna', 'grid', 'autogluon')
      - param_config: Optional parameter configuration
      
    Returns:
      - dict: Optimization results
    """
    # Load default strategy parameters
    default_params = get_strategy_parameters(GlobalSettings.ACTIVE_STRATEGY)
    
    # Default parameter space
    default_param_space = {
        'stop_loss': {
            'type': 'float',
            'low': 0.01,
            'high': 0.05
        },
        'take_profit': {
            'type': 'float',
            'low': 0.02,
            'high': 0.10
        },
        'risk_factor': {
            'type': 'float',
            'low': 0.005,
            'high': 0.02
        },
        'vol_window': {
            'type': 'int',
            'low': 10,
            'high': 30
        },
        'rsi_period': {
            'type': 'int',
            'low': 7,
            'high': 21
        },
        'ml_weight': {
            'type': 'float',
            'low': 0.2,
            'high': 0.6
        },
        'sentiment_weight': {
            'type': 'float',
            'low': 0.1,
            'high': 0.5
        },
        'tech_weight': {
            'type': 'float',
            'low': 0.2,
            'high': 0.6
        }
    }
    
    # Default grid search parameters
    default_param_grid = {
        'stop_loss': [0.02, 0.03, 0.04],
        'take_profit': [0.04, 0.06, 0.08],
        'risk_factor': [0.005, 0.01, 0.015],
        'vol_window': [15, 20, 25],
        'rsi_period': [9, 14, 21],
        'ml_weight': [0.3, 0.4, 0.5],
        'sentiment_weight': [0.2, 0.3, 0.4],
        'tech_weight': [0.2, 0.3, 0.4]
    }
    
    # Use provided param config or defaults
    if param_config is None:
        if optimization_type == 'optuna':
            param_config = default_param_space
        else:
            param_config = default_param_grid
    
    # Default test symbols
    symbols = ['AAPL', 'MSFT', 'AMZN', 'GOOGL', 'META']
    
    # Default date range (1 year)
    end_date = datetime.date.today().strftime('%Y-%m-%d')
    start_date = (datetime.date.today() - datetime.timedelta(days=365)).strftime('%Y-%m-%d')
    
    # Run optimization
    if optimization_type == 'optuna':
        return optimize_parameters(
            symbols=symbols,
            start_date=start_date,
            end_date=end_date,
            param_space=param_config,
            n_trials=50,
            objective_metric='sharpe'
        )
    
    elif optimization_type == 'grid':
        return grid_search(
            param_grid=param_config,
            symbols=symbols,
            start_date=start_date,
            end_date=end_date,
            n_jobs=4,
            objective_metric='sharpe'
        )
    
    elif optimization_type == 'autogluon':
        return optimize_autogluon(
            param_grid=param_config,
            n_trials=20,
            time_limit=600,
            metric='sharpe'
        )
    
    else:
        logger.error(f"Unknown optimization type: {optimization_type}")
        return None

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Parse command line arguments
    import argparse
    
    parser = argparse.ArgumentParser(description='Parameter Optimization')
    parser.add_argument('--method', choices=['optuna', 'grid', 'autogluon'], default='optuna',
                      help='Optimization method')
    parser.add_argument('--trials', type=int, default=50,
                      help='Number of trials for Optuna or AutoGluon')
    parser.add_argument('--metric', default='sharpe',
                      help='Metric to optimize')
    
    args = parser.parse_args()
    
    # Run optimization
    print(f"Running {args.method} optimization for {args.metric} with {args.trials} trials")
    
    if args.method == 'optuna':
        # Sample a smaller parameter space for testing
        param_space = {
            'stop_loss': {
                'type': 'float',
                'low': 0.02,
                'high': 0.04
            },
            'take_profit': {
                'type': 'float',
                'low': 0.03,
                'high': 0.08
            },
            'risk_factor': {
                'type': 'float',
                'low': 0.007,
                'high': 0.015
            }
        }
        
        results = optimize_parameters(
            symbols=['AAPL', 'MSFT', 'GOOGL'],
            start_date='2022-06-01',
            end_date='2023-06-01',
            param_space=param_space,
            n_trials=args.trials,
            objective_metric=args.metric
        )
        
        print(f"Best parameters: {results['best_params']}")
        print(f"Best metrics: {results['best_metrics']}")
    
    elif args.method == 'grid':
        # Sample a smaller grid for testing
        param_grid = {
            'stop_loss': [0.02, 0.03],
            'take_profit': [0.04, 0.06],
            'risk_factor': [0.01, 0.015]
        }
        
        results_df = grid_search(
            param_grid=param_grid,
            symbols=['AAPL', 'MSFT', 'GOOGL'],
            start_date='2022-06-01',
            end_date='2023-06-01',
            n_jobs=2,
            objective_metric=args.metric
        )
        
        print("Top 5 parameter combinations:")
        print(results_df.head())
    
    elif args.method == 'autogluon':
        # Sample a smaller grid for testing
        param_grid = {
            'stop_loss': [0.02, 0.03, 0.04],
            'take_profit': [0.04, 0.06, 0.08],
            'risk_factor': [0.005, 0.01, 0.015]
        }
        
        best_params = optimize_autogluon(
            param_grid=param_grid,
            n_trials=args.trials,
            time_limit=300,
            metric=args.metric
        )
        
        print(f"Best parameters: {best_params}")