import backtrader as bt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
import os
import json
from modules.asset_selector import select_top_assets
from modules.data_acquisition import fetch_price_data
from modules.strategy import AdaptiveSentimentStrategy

class BacktestEngine:
    def __init__(self, initial_cash=100000, benchmark_symbol='^GSPC'):
        self.initial_cash = initial_cash
        self.benchmark_symbol = benchmark_symbol
        self.cerebro = bt.Cerebro()
        self.cerebro.broker.setcash(initial_cash)

    def add_strategy(self, strategy_class, **kwargs):
        """Add a strategy with parameters to the backtest engine"""
        self.cerebro.addstrategy(strategy_class, **kwargs)
        return self

    def add_data(self, symbols, start_date, end_date, data_dir='data'):
        """Add data feeds for multiple symbols"""
        os.makedirs(data_dir, exist_ok=True)
        
        for symbol in symbols:
            try:
                data = fetch_price_data(symbol, start_date, end_date)
                data_file = f'{data_dir}/{symbol}_prices.csv'
                data.to_csv(data_file)
                
                data_feed = bt.feeds.YahooFinanceCSVData(
                    dataname=data_file,
                    fromdate=datetime.datetime.strptime(start_date, '%Y-%m-%d'),
                    todate=datetime.datetime.strptime(end_date, '%Y-%m-%d'),
                    reverse=False
                )
                self.cerebro.adddata(data_feed, name=symbol)
                print(f"Added data for {symbol}")
            except Exception as e:
                print(f"Error adding data for {symbol}: {e}")
        
        return self
    
    def add_analyzers(self):
        """Add standard analyzers for performance metrics"""
        self.cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
        self.cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
        self.cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
        self.cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')
        self.cerebro.addanalyzer(bt.analyzers.SQN, _name='sqn')
        return self
    
    def run(self):
        """Run the backtest and return the results"""
        results = self.cerebro.run()
        self.strategy = results[0]
        return self
    
    def get_metrics(self):
        """Extract performance metrics from analyzers"""
        metrics = {
            'final_value': self.cerebro.broker.getvalue(),
            'pnl': self.cerebro.broker.getvalue() - self.initial_cash,
            'pnl_percent': (self.cerebro.broker.getvalue() / self.initial_cash - 1) * 100,
            'sharpe': getattr(self.strategy.analyzers.sharpe, 'ratio', 0),
            'max_drawdown': getattr(self.strategy.analyzers.drawdown, 'maxdrawdown', 0) * 100,
            'sqn': getattr(self.strategy.analyzers.sqn, 'sqn', 0),
        }
        
        # Add trade metrics if trades were made
        if hasattr(self.strategy.analyzers.trades, 'rets'):
            trades = self.strategy.analyzers.trades.rets
            metrics.update({
                'total_trades': trades.get('total', {}).get('total', 0),
                'win_rate': trades.get('won', {}).get('total', 0) / max(1, trades.get('total', {}).get('total', 1)) * 100,
                'avg_trade_pnl': trades.get('pnl', {}).get('net', {}).get('average', 0),
            })
        
        return metrics
    
    def get_benchmark_performance(self, start_date, end_date):
        """Fetch benchmark data and calculate performance"""
        import yfinance as yf
        benchmark = yf.download(self.benchmark_symbol, start=start_date, end=end_date)['Close']
        benchmark_return = (benchmark.iloc[-1] / benchmark.iloc[0] - 1) * 100
        return {
            'symbol': self.benchmark_symbol,
            'start_price': benchmark.iloc[0],
            'end_price': benchmark.iloc[-1],
            'return_percent': benchmark_return
        }
    
    def plot(self, filename=None):
        """Generate performance plots"""
        figs = self.cerebro.plot(style='candlestick', barup='green', bardown='red', 
                             volup='green', voldown='red', 
                             width=12, height=8)
        
        if filename:
            for i, fig in enumerate(figs):
                fig[0].savefig(f'{filename}_{i}.png')
        
        return self
    
    def save_results(self, filename='backtest_results.json'):
        """Save backtest metrics to file"""
        with open(filename, 'w') as f:
            json.dump(self.get_metrics(), f, indent=4)
        return self

def parameter_sweep(param_grid, backtest_func):
    """
    Run backtests for all combinations of parameters in param_grid
    param_grid: Dict of parameter names to lists of values
    backtest_func: Function that takes parameters and returns metrics
    """
    import itertools
    
    # Generate all combinations of parameters
    param_names = list(param_grid.keys())
    param_values = list(itertools.product(*[param_grid[name] for name in param_names]))
    
    results = []
    for values in param_values:
        params = dict(zip(param_names, values))
        print(f"Testing parameters: {params}")
        
        metrics = backtest_func(**params)
        params.update(metrics)
        results.append(params)
    
    return pd.DataFrame(results)

def run_full_backtest(start_date, end_date, assets=None, n_assets=10, strategy_params=None):
    """
    Run a complete backtest with the specified parameters
    """
    if strategy_params is None:
        strategy_params = {}
    
    # Select assets if not provided
    if assets is None:
        assets = select_top_assets(n=n_assets)
    
    # Create and run backtest engine
    engine = BacktestEngine(initial_cash=100000)
    engine.add_strategy(AdaptiveSentimentStrategy, **strategy_params)
    engine.add_data(assets, start_date, end_date)
    engine.add_analyzers()
    engine.run()
    
    # Get metrics and benchmark
    metrics = engine.get_metrics()
    benchmark = engine.get_benchmark_performance(start_date, end_date)
    
    # Generate plots
    engine.plot(filename=f'backtest_{start_date}_to_{end_date}')
    
    # Print summary
    print("\n=== BACKTEST RESULTS ===")
    print(f"Period: {start_date} to {end_date}")
    print(f"Assets: {', '.join(assets)}")
    print(f"Final Portfolio Value: ${metrics['final_value']:.2f}")
    print(f"PnL: ${metrics['pnl']:.2f} ({metrics['pnl_percent']:.2f}%)")
    print(f"Benchmark ({benchmark['symbol']}): {benchmark['return_percent']:.2f}%")
    print(f"Alpha: {metrics['pnl_percent'] - benchmark['return_percent']:.2f}%")
    print(f"Sharpe Ratio: {metrics['sharpe']:.2f}")
    print(f"Max Drawdown: {metrics['max_drawdown']:.2f}%")
    print(f"System Quality Number: {metrics['sqn']:.2f}")
    
    if 'total_trades' in metrics:
        print(f"Total Trades: {metrics['total_trades']}")
        print(f"Win Rate: {metrics['win_rate']:.2f}%")
        print(f"Average Trade PnL: ${metrics['avg_trade_pnl']:.2f}")
    
    # Save results
    engine.save_results()
    
    return metrics, benchmark, engine

if __name__ == '__main__':
    # Example usage
    run_full_backtest('2022-01-01', '2023-01-01', n_assets=5)
    
    # Example parameter sweep
    param_grid = {
        'stop_loss': [0.02, 0.03, 0.04],
        'take_profit': [0.04, 0.06, 0.08],
        'risk_factor': [0.005, 0.01, 0.015]
    }
    
    def backtest_with_params(stop_loss, take_profit, risk_factor):
        engine = BacktestEngine()
        engine.add_strategy(AdaptiveSentimentStrategy, 
                            stop_loss=stop_loss, 
                            take_profit=take_profit, 
                            risk_factor=risk_factor)
        engine.add_data(['AAPL', 'MSFT', 'GOOGL'], '2022-01-01', '2022-06-30')
        engine.add_analyzers()
        engine.run()
        return engine.get_metrics()
    
    results_df = parameter_sweep(param_grid, backtest_with_params)
    results_df.to_csv('parameter_sweep_results.csv', index=False)
    print("Best parameters by Sharpe ratio:")
    print(results_df.loc[results_df['sharpe'].idxmax()])