# modules/backtest.py
import backtrader as bt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
import os
import json
import multiprocessing
import logging
import itertools
from functools import partial
import time
from modules.asset_selector import select_top_assets
from modules.data_acquisition import fetch_price_data, batch_fetch_price_data
from modules.strategy import AdaptiveSentimentStrategy
from modules.risk_manager import RiskManager

# Configure logging
logger = logging.getLogger(__name__)

class BacktestEngine:
    """Enhanced backtesting engine with improved error handling and performance"""
    
    def __init__(self, initial_cash=100000, benchmark_symbol='^GSPC', data_dir='data', 
                risk_manager=None):
        """
        Initialize the backtest engine
        
        Parameters:
          - initial_cash: Starting cash for the backtest
          - benchmark_symbol: Symbol to use as benchmark
          - data_dir: Directory to store price data
          - risk_manager: Optional RiskManager instance
        """
        self.initial_cash = initial_cash
        self.benchmark_symbol = benchmark_symbol
        self.data_dir = data_dir
        self.cerebro = bt.Cerebro()
        self.cerebro.broker.setcash(initial_cash)
        self.risk_manager = risk_manager or RiskManager()
        self.data_feeds = {}
        self.symbols = []
        
        # Store metadata about the backtest
        self.metadata = {
            'start_time': datetime.datetime.now().isoformat(),
            'initial_cash': initial_cash,
            'benchmark': benchmark_symbol,
        }
        
        # Ensure data directory exists
        os.makedirs(data_dir, exist_ok=True)

    def add_strategy(self, strategy_class, **kwargs):
        """Add a strategy with parameters to the backtest engine"""
        # Add the risk manager to the strategy parameters
        kwargs['risk_manager'] = self.risk_manager
        
        # Log strategy details
        logger.info(f"Adding strategy: {strategy_class.__name__}")
        logger.info(f"Strategy parameters: {kwargs}")
        
        self.cerebro.addstrategy(strategy_class, **kwargs)
        
        # Update metadata
        self.metadata['strategy'] = {
            'name': strategy_class.__name__,
            'parameters': {k: str(v) if not isinstance(v, (int, float, bool, str)) else v 
                          for k, v in kwargs.items() if k != 'risk_manager'}
        }
        
        return self

    def add_data(self, symbols, start_date, end_date, data_dir=None):
        """Add data feeds for multiple symbols with enhanced error handling"""
        data_dir = data_dir or self.data_dir
        os.makedirs(data_dir, exist_ok=True)
        
        # Store symbols
        self.symbols = symbols.copy() if isinstance(symbols, list) else [symbols]
        
        # Update metadata
        self.metadata['data'] = {
            'symbols': self.symbols,
            'start_date': start_date,
            'end_date': end_date
        }
        
        # Convert string dates to datetime
        start_dt = pd.to_datetime(start_date).to_pydatetime()
        end_dt = pd.to_datetime(end_date).to_pydatetime()
        
        # Fetch data for all symbols in batches
        logger.info(f"Fetching price data for {len(symbols)} symbols")
        
        try:
            # Use batch fetching for efficiency
            if len(symbols) > 1:
                data_dict = batch_fetch_price_data(symbols, start_date, end_date, batch_size=5)
            else:
                data_dict = {symbols[0]: fetch_price_data(symbols[0], start_date, end_date)}
            
            # Process each symbol's data
            successful_symbols = []
            
            for symbol, data in data_dict.items():
                if data is None or data.empty:
                    logger.warning(f"No data available for {symbol}")
                    continue
                
                try:
                    # Save to CSV
                    data_file = f'{data_dir}/{symbol}_prices.csv'
                    data.to_csv(data_file)
                    
                    # Create Backtrader data feed
                    data_feed = bt.feeds.GenericCSVData(
                        dataname=data_file,
                        fromdate=start_dt,
                        todate=end_dt,
                        nullvalue=0.0,
                        dtformat=('%Y-%m-%d'),
                        datetime=0,
                        open=1,
                        high=2,
                        low=3,
                        close=4,
                        volume=5,
                        openinterest=-1
                    )
                    
                    # Add the data feed to cerebro with a name
                    self.cerebro.adddata(data_feed, name=symbol)
                    self.data_feeds[symbol] = data_feed
                    successful_symbols.append(symbol)
                    logger.info(f"Added data for {symbol}")
                    
                except Exception as e:
                    logger.error(f"Error processing data for {symbol}: {e}")
            
            # Update the list of symbols to only include successful ones
            self.symbols = successful_symbols
            if not successful_symbols:
                raise ValueError("No data was successfully loaded for any symbol")
                
        except Exception as e:
            logger.error(f"Error in add_data: {e}")
            raise
        
        return self
    
    def add_analyzers(self):
        """Add comprehensive analyzers for performance metrics"""
        self.cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe', riskfreerate=0.02)
        self.cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
        self.cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
        self.cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')
        self.cerebro.addanalyzer(bt.analyzers.SQN, _name='sqn')
        self.cerebro.addanalyzer(bt.analyzers.TimeReturn, _name='timereturn')
        self.cerebro.addanalyzer(bt.analyzers.VWR, _name='vwr')
        self.cerebro.addanalyzer(bt.analyzers.AnnualReturn, _name='annual')
        self.cerebro.addanalyzer(bt.analyzers.PeriodStats, _name='periodstats')
        self.cerebro.addanalyzer(bt.analyzers.PyFolio, _name='pyfolio')
        
        return self
    
    def run(self):
        """Run the backtest with exception handling"""
        if not self.symbols:
            raise ValueError("No data has been added to the backtest")
        
        logger.info("Starting backtest")
        start_time = time.time()
        
        try:
            results = self.cerebro.run()
            self.strategy = results[0]
            
            # Update metadata
            self.metadata['end_time'] = datetime.datetime.now().isoformat()
            self.metadata['duration_seconds'] = time.time() - start_time
            
            logger.info(f"Backtest completed in {self.metadata['duration_seconds']:.2f} seconds")
            
        except Exception as e:
            logger.error(f"Error during backtest: {e}")
            raise
        
        return self
    
    def get_metrics(self):
        """Extract comprehensive performance metrics from analyzers"""
        if not hasattr(self, 'strategy'):
            raise ValueError("Backtest has not been run yet")
        
        metrics = {
            'final_value': self.cerebro.broker.getvalue(),
            'pnl': self.cerebro.broker.getvalue() - self.initial_cash,
            'pnl_percent': (self.cerebro.broker.getvalue() / self.initial_cash - 1) * 100,
            'sharpe': getattr(self.strategy.analyzers.sharpe, 'ratio', 0),
            'max_drawdown': getattr(self.strategy.analyzers.drawdown, 'maxdrawdown', 0) * 100,
            'max_drawdown_length': getattr(self.strategy.analyzers.drawdown, 'maxdrawdownperiod', 0),
            'sqn': getattr(self.strategy.analyzers.sqn, 'sqn', 0),
            'vwr': getattr(self.strategy.analyzers.vwr, 'vwr', 0),
        }
        
        # Add period stats
        if hasattr(self.strategy.analyzers, 'periodstats'):
            period_stats = self.strategy.analyzers.periodstats.get_analysis()
            metrics.update({
                'average_return': period_stats.get('average', 0),
                'stddev_return': period_stats.get('stddev', 0),
                'positive_periods': period_stats.get('positive', 0),
                'negative_periods': period_stats.get('negative', 0)
            })
        
        # Add annual returns if available
        if hasattr(self.strategy.analyzers, 'annual'):
            annual_returns = self.strategy.analyzers.annual.get_analysis()
            metrics['annual_returns'] = {str(year): return_val for year, return_val in annual_returns.items()}
        
        # Add trade metrics if trades were made
        if hasattr(self.strategy.analyzers.trades, 'rets') and self.strategy.analyzers.trades.rets:
            trades = self.strategy.analyzers.trades.rets
            
            # Handle case where trades is None or doesn't have expected attributes
            if trades is not None:
                won_trades = trades.get('won', {}).get('total', 0)
                total_trades = trades.get('total', {}).get('total', 0)
                
                metrics.update({
                    'total_trades': total_trades,
                    'win_rate': won_trades / max(1, total_trades) * 100 if total_trades else 0,
                    'avg_trade_pnl': trades.get('pnl', {}).get('net', {}).get('average', 0),
                    'avg_winning_trade': trades.get('won', {}).get('pnl', {}).get('average', 0),
                    'avg_losing_trade': trades.get('lost', {}).get('pnl', {}).get('average', 0),
                    'largest_winning_trade': trades.get('won', {}).get('pnl', {}).get('max', 0),
                    'largest_losing_trade': trades.get('lost', {}).get('pnl', {}).get('max', 0),
                })
        
        return metrics
    
    def get_benchmark_performance(self, start_date, end_date):
        """Fetch benchmark data and calculate performance with error handling"""
        try:
            benchmark_data = fetch_price_data(self.benchmark_symbol, start_date, end_date)
            
            if benchmark_data is None or benchmark_data.empty:
                logger.warning(f"No benchmark data available for {self.benchmark_symbol}")
                return {
                    'symbol': self.benchmark_symbol,
                    'start_price': None,
                    'end_price': None,
                    'return_percent': 0
                }
            
            benchmark_return = (benchmark_data['Close'].iloc[-1] / benchmark_data['Close'].iloc[0] - 1) * 100
            
            return {
                'symbol': self.benchmark_symbol,
                'start_price': benchmark_data['Close'].iloc[0],
                'end_price': benchmark_data['Close'].iloc[-1],
                'return_percent': benchmark_return,
                'annualized_return': self._calculate_annualized_return(
                    benchmark_data['Close'].iloc[0],
                    benchmark_data['Close'].iloc[-1],
                    len(benchmark_data)
                )
            }
            
        except Exception as e:
            logger.error(f"Error getting benchmark performance: {e}")
            return {
                'symbol': self.benchmark_symbol,
                'error': str(e),
                'return_percent': 0
            }
    
    def _calculate_annualized_return(self, start_price, end_price, days):
        """Calculate annualized return"""
        if days <= 0 or start_price <= 0:
            return 0
        
        total_return = end_price / start_price
        years = days / 252  # Assuming 252 trading days in a year
        
        if years <= 0:
            return 0
            
        annualized = (total_return ** (1 / years)) - 1
        return annualized * 100  # Convert to percentage
    
    def plot(self, filename=None, show=False):
        """Generate comprehensive performance plots with error handling"""
        if not hasattr(self, 'strategy'):
            raise ValueError("Backtest has not been run yet")
        
        try:
            # Create a output directory if it doesn't exist
            plot_dir = 'plots'
            if filename:
                os.makedirs(os.path.dirname(os.path.join(plot_dir, filename)), exist_ok=True)
            
            # Generate plots
            figs = self.cerebro.plot(style='candlestick', barup='green', bardown='red', 
                                volup='green', voldown='red', 
                                width=16, height=9, dpi=100)
            
            if filename:
                for i, fig in enumerate(figs):
                    fig_path = f'{plot_dir}/{filename}_fig{i}.png'
                    fig[0].savefig(fig_path)
                    logger.info(f"Saved plot to {fig_path}")
            
            if show:
                plt.show()
            
        except Exception as e:
            logger.error(f"Error generating plots: {e}")
        
        return self
    
    def save_results(self, filename='backtest_results.json'):
        """Save backtest metrics and metadata to file"""
        if not hasattr(self, 'strategy'):
            raise ValueError("Backtest has not been run yet")
        
        try:
            # Combine metrics and metadata
            results = {
                'metrics': self.get_metrics(),
                'metadata': self.metadata
            }
            
            # Add benchmark data if available
            if 'data' in self.metadata:
                start_date = self.metadata['data']['start_date']
                end_date = self.metadata['data']['end_date']
                results['benchmark'] = self.get_benchmark_performance(start_date, end_date)
            
            # Save to file
            with open(filename, 'w') as f:
                json.dump(results, f, indent=4, default=str)
            
            logger.info(f"Results saved to {filename}")
            
        except Exception as e:
            logger.error(f"Error saving results: {e}")
        
        return self
    
    def generate_tear_sheet(self, filename='tear_sheet.html'):
        """Generate an HTML tear sheet with performance metrics and visualizations"""
        if not hasattr(self, 'strategy'):
            raise ValueError("Backtest has not been run yet")
        
        try:
            # Get performance metrics
            metrics = self.get_metrics()
            
            # Get benchmark data
            if 'data' in self.metadata:
                start_date = self.metadata['data']['start_date']
                end_date = self.metadata['data']['end_date']
                benchmark = self.get_benchmark_performance(start_date, end_date)
            else:
                benchmark = {'symbol': self.benchmark_symbol, 'return_percent': 0}
            
            # Extract returns from pyfolio analyzer if available
            returns_data = None
            if hasattr(self.strategy.analyzers, 'pyfolio'):
                returns, positions, transactions, gross_lev = self.strategy.analyzers.pyfolio.get_pf_items()
                returns_data = returns
            
            # Generate HTML content
            html_content = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>Backtest Tear Sheet</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 20px; }}
                    .container {{ max-width: 1200px; margin: 0 auto; }}
                    .header {{ text-align: center; margin-bottom: 30px; }}
                    .section {{ margin-bottom: 40px; }}
                    .metrics-grid {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 15px; }}
                    .metric-card {{ border: 1px solid #ddd; border-radius: 5px; padding: 15px; }}
                    .metric-value {{ font-size: 24px; font-weight: bold; margin: 10px 0; }}
                    .metric-name {{ color: #555; }}
                    table {{ width: 100%; border-collapse: collapse; }}
                    th, td {{ border: 1px solid #ddd; padding: 10px; text-align: left; }}
                    th {{ background-color: #f2f2f2; }}
                    .positive {{ color: green; }}
                    .negative {{ color: red; }}
                </style>
            </head>
            <body>
                <div class="container">
                    <div class="header">
                        <h1>Backtest Performance Report</h1>
                        <p>Strategy: {self.metadata.get('strategy', {}).get('name', 'Unknown')}</p>
                        <p>Period: {self.metadata.get('data', {}).get('start_date', '')} to {self.metadata.get('data', {}).get('end_date', '')}</p>
                        <p>Symbols: {', '.join(self.symbols)}</p>
                    </div>
                    
                    <div class="section">
                        <h2>Summary</h2>
                        <div class="metrics-grid">
                            <div class="metric-card">
                                <div class="metric-name">Final Portfolio Value</div>
                                <div class="metric-value">${metrics['final_value']:.2f}</div>
                            </div>
                            <div class="metric-card">
                                <div class="metric-name">Total Return</div>
                                <div class="metric-value {('positive' if metrics['pnl_percent'] >= 0 else 'negative')}">{metrics['pnl_percent']:.2f}%</div>
                            </div>
                            <div class="metric-card">
                                <div class="metric-name">Benchmark Return ({benchmark['symbol']})</div>
                                <div class="metric-value {('positive' if benchmark['return_percent'] >= 0 else 'negative')}">{benchmark['return_percent']:.2f}%</div>
                            </div>
                            <div class="metric-card">
                                <div class="metric-name">Alpha</div>
                                <div class="metric-value {('positive' if (metrics['pnl_percent'] - benchmark['return_percent']) >= 0 else 'negative')}">{metrics['pnl_percent'] - benchmark['return_percent']:.2f}%</div>
                            </div>
                            <div class="metric-card">
                                <div class="metric-name">Sharpe Ratio</div>
                                <div class="metric-value">{metrics['sharpe']:.2f}</div>
                            </div>
                            <div class="metric-card">
                                <div class="metric-name">Max Drawdown</div>
                                <div class="metric-value negative">{metrics['max_drawdown']:.2f}%</div>
                            </div>
                        </div>
                    </div>
                    
                    <div class="section">
                        <h2>Trade Statistics</h2>
                        <table>
                            <tr>
                                <th>Metric</th>
                                <th>Value</th>
                            </tr>
                            <tr>
                                <td>Total Trades</td>
                                <td>{metrics.get('total_trades', 'N/A')}</td>
                            </tr>
                            <tr>
                                <td>Win Rate</td>
                                <td>{metrics.get('win_rate', 'N/A'):.2f}%</td>
                            </tr>
                            <tr>
                                <td>Average Trade P&L</td>
                                <td>${metrics.get('avg_trade_pnl', 'N/A'):.2f}</td>
                            </tr>
                            <tr>
                                <td>Average Winning Trade</td>
                                <td>${metrics.get('avg_winning_trade', 'N/A'):.2f}</td>
                            </tr>
                            <tr>
                                <td>Average Losing Trade</td>
                                <td>${metrics.get('avg_losing_trade', 'N/A'):.2f}</td>
                            </tr>
                            <tr>
                                <td>Largest Winning Trade</td>
                                <td>${metrics.get('largest_winning_trade', 'N/A'):.2f}</td>
                            </tr>
                            <tr>
                                <td>Largest Losing Trade</td>
                                <td>${metrics.get('largest_losing_trade', 'N/A'):.2f}</td>
                            </tr>
                            <tr>
                                <td>System Quality Number (SQN)</td>
                                <td>{metrics.get('sqn', 'N/A'):.2f}</td>
                            </tr>
                        </table>
                    </div>
                    
                    <div class="section">
                        <h2>Advanced Metrics</h2>
                        <table>
                            <tr>
                                <th>Metric</th>
                                <th>Value</th>
                            </tr>
                            <tr>
                                <td>Variance-Weighted Return (VWR)</td>
                                <td>{metrics.get('vwr', 'N/A'):.4f}</td>
                            </tr>
                            <tr>
                                <td>Max Drawdown Duration</td>
                                <td>{metrics.get('max_drawdown_length', 'N/A')} days</td>
                            </tr>
                            <tr>
                                <td>Average Period Return</td>
                                <td>{metrics.get('average_return', 'N/A'):.4f}</td>
                            </tr>
                            <tr>
                                <td>StdDev of Period Returns</td>
                                <td>{metrics.get('stddev_return', 'N/A'):.4f}</td>
                            </tr>
                            <tr>
                                <td>Positive Periods</td>
                                <td>{metrics.get('positive_periods', 'N/A')}</td>
                            </tr>
                            <tr>
                                <td>Negative Periods</td>
                                <td>{metrics.get('negative_periods', 'N/A')}</td>
                            </tr>
                        </table>
                    </div>
                    
                    <div class="section">
                        <h2>Strategy Parameters</h2>
                        <table>
                            <tr>
                                <th>Parameter</th>
                                <th>Value</th>
                            </tr>
            """
            
            # Add strategy parameters
            for param, value in self.metadata.get('strategy', {}).get('parameters', {}).items():
                html_content += f"""
                            <tr>
                                <td>{param}</td>
                                <td>{value}</td>
                            </tr>
                """
            
            html_content += """
                        </table>
                    </div>
                </div>
            </body>
            </html>
            """
            
            # Save HTML file
            with open(filename, 'w') as f:
                f.write(html_content)
            
            logger.info(f"Tear sheet saved to {filename}")
            
        except Exception as e:
            logger.error(f"Error generating tear sheet: {e}")
        
        return self

def _run_single_backtest(params, backtest_func):
    """Helper function to run a single backtest with parameters"""
    try:
        logger.info(f"Running backtest with parameters: {params}")
        result = backtest_func(**params)
        return {**params, **result}
    except Exception as e:
        logger.error(f"Error in backtest with params {params}: {e}")
        return {**params, 'error': str(e)}

def parameter_sweep(param_grid, backtest_func, n_jobs=None):
    """
    Run parallel backtests for all combinations of parameters in param_grid
    
    Parameters:
      - param_grid: Dict of parameter names to lists of values
      - backtest_func: Function that takes parameters and returns metrics
      - n_jobs: Number of parallel jobs (None for all available)
      
    Returns:
      - pd.DataFrame with results for all parameter combinations
    """
    # Generate all combinations of parameters
    param_names = list(param_grid.keys())
    param_values = list(itertools.product(*[param_grid[name] for name in param_names]))
    params_list = [dict(zip(param_names, values)) for values in param_values]
    
    logger.info(f"Running parameter sweep with {len(params_list)} combinations")
    start_time = time.time()
    
    if n_jobs is None or n_jobs > 1:
        # Use process pool for parallel execution
        with multiprocessing.Pool(processes=n_jobs) as pool:
            results = pool.map(partial(_run_single_backtest, backtest_func=backtest_func), params_list)
    else:
        # Run sequentially
        results = []
        for params in params_list:
            results.append(_run_single_backtest(params, backtest_func))
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    # Log elapsed time
    elapsed = time.time() - start_time
    logger.info(f"Parameter sweep completed in {elapsed:.2f} seconds")
    
    return results_df

def run_full_backtest(start_date, end_date, assets=None, n_assets=10, strategy_params=None,
                     benchmark_symbol='^GSPC', initial_cash=100000, risk_manager=None):
    """
    Run a complete backtest with the specified parameters and enhanced reporting
    
    Parameters:
      - start_date: Start date for backtest data
      - end_date: End date for backtest data
      - assets: List of asset symbols to include, or None to select dynamically
      - n_assets: Number of assets to select if assets is None
      - strategy_params: Parameters to pass to the strategy
      - benchmark_symbol: Symbol to use as benchmark
      - initial_cash: Starting cash for the backtest
      - risk_manager: Optional RiskManager instance
      
    Returns:
      - tuple: (metrics, benchmark, engine)
    """
    if strategy_params is None:
        strategy_params = {}
    
    # Select assets if not provided
    if assets is None:
        try:
            assets = select_top_assets(n=n_assets)
            logger.info(f"Selected {len(assets)} assets dynamically")
        except Exception as e:
            logger.error(f"Error selecting assets: {e}")
            # Fallback to some default assets
            assets = ["AAPL", "MSFT", "AMZN", "GOOGL", "META"][:n_assets]
            logger.info(f"Using fallback assets: {assets}")
    
    # Create and setup risk manager if not provided
    if risk_manager is None:
        risk_manager = RiskManager(
            max_position_size=0.2,
            max_portfolio_risk=0.05,
            max_drawdown_limit=0.15,
            position_sizing_method='volatility'
        )
    
    # Create and run backtest engine
    logger.info(f"Starting backtest from {start_date} to {end_date} with {len(assets)} assets")
    start_time = time.time()
    
    try:
        engine = BacktestEngine(
            initial_cash=initial_cash,
            benchmark_symbol=benchmark_symbol,
            risk_manager=risk_manager
        )
        
        engine.add_strategy(AdaptiveSentimentStrategy, **strategy_params)
        engine.add_data(assets, start_date, end_date)
        engine.add_analyzers()
        engine.run()
        
        # Get metrics and benchmark
        metrics = engine.get_metrics()
        benchmark = engine.get_benchmark_performance(start_date, end_date)
        
        # Generate plots and reports
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        engine.plot(filename=f'backtest_{timestamp}')
        engine.save_results(filename=f'results/backtest_{timestamp}.json')
        engine.generate_tear_sheet(filename=f'reports/tear_sheet_{timestamp}.html')
        
        # Calculate elapsed time
        elapsed = time.time() - start_time
        
        # Print summary
        print("\n" + "="*50)
        print("BACKTEST RESULTS")
        print("="*50)
        print(f"Period: {start_date} to {end_date}")
        print(f"Assets: {', '.join(assets)}")
        print(f"Execution time: {elapsed:.2f} seconds")
        print(f"\nFinal Portfolio Value: ${metrics['final_value']:.2f}")
        print(f"PnL: ${metrics['pnl']:.2f} ({metrics['pnl_percent']:.2f}%)")
        print(f"Benchmark ({benchmark['symbol']}): {benchmark['return_percent']:.2f}%")
        print(f"Alpha: {metrics['pnl_percent'] - benchmark['return_percent']:.2f}%")
        print(f"Sharpe Ratio: {metrics['sharpe']:.2f}")
        print(f"Max Drawdown: {metrics['max_drawdown']:.2f}%")
        print(f"System Quality Number: {metrics['sqn']:.2f}")
        
        if 'total_trades' in metrics:
            print(f"\nTotal Trades: {metrics['total_trades']}")
            print(f"Win Rate: {metrics['win_rate']:.2f}%")
            print(f"Average Trade PnL: ${metrics['avg_trade_pnl']:.2f}")
        
        print("\nDetailed results saved to:")
        print(f"  - JSON: results/backtest_{timestamp}.json")
        print(f"  - HTML: reports/tear_sheet_{timestamp}.html")
        print(f"  - Plots: plots/backtest_{timestamp}_*.png")
        print("="*50)
        
        return metrics, benchmark, engine
        
    except Exception as e:
        logger.error(f"Error running full backtest: {e}")
        raise

if __name__ == '__main__':
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler("backtest.log"),
            logging.StreamHandler()
        ]
    )
    
    # Create output directories
    os.makedirs('data', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    os.makedirs('reports', exist_ok=True)
    os.makedirs('plots', exist_ok=True)
    
    # Example usage
    run_full_backtest('2022-01-01', '2023-01-01', n_assets=5)
    
    # Example parameter sweep
    param_grid = {
        'stop_loss': [0.02, 0.03, 0.04],
        'take_profit': [0.04, 0.06, 0.08],
        'risk_factor': [0.005, 0.01, 0.015]
    }
    
    def backtest_with_params(stop_loss, take_profit, risk_factor):
        # Create a risk manager with the parameters
        risk_manager = RiskManager(
            stop_loss_pct=stop_loss,
            take_profit_pct=take_profit,
            position_sizing_method='volatility'
        )
        
        # Create engine
        engine = BacktestEngine(risk_manager=risk_manager)
        
        # Setup and run
        engine.add_strategy(AdaptiveSentimentStrategy, 
                           stop_loss=stop_loss, 
                           take_profit=take_profit, 
                           risk_factor=risk_factor)
        engine.add_data(['AAPL', 'MSFT', 'GOOGL'], '2022-01-01', '2022-06-30')
        engine.add_analyzers()
        engine.run()
        
        return engine.get_metrics()
    
    # Run the parameter sweep with parallelization
    results_df = parameter_sweep(param_grid, backtest_with_params, n_jobs=4)
    results_df.to_csv('results/parameter_sweep_results.csv', index=False)
    
    # Print the best parameters
    print("\nBest parameters by Sharpe ratio:")
    best_row = results_df.loc[results_df['sharpe'].idxmax()]
    print(best_row)
    
    # Run a final backtest with the best parameters
    best_params = {
        'stop_loss': best_row['stop_loss'],
        'take_profit': best_row['take_profit'],
        'risk_factor': best_row['risk_factor']
    }
    
    print("\nRunning final backtest with best parameters:")
    run_full_backtest('2022-01-01', '2023-01-01', n_assets=5, strategy_params=best_params)