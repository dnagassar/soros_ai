# scheduler.py
"""
Scheduler module for automated trading system execution
"""
import schedule
import time
import logging
import datetime
import os
import sys
import json
import pandas as pd
import argparse
import threading
from queue import Queue
import signal

# Add project root to path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

# Import project modules
from config import (
    SystemConfig, 
    GlobalSettings, 
    get_strategy_parameters,
    get_risk_parameters
)
from modules.data_acquisition import fetch_price_data, batch_fetch_price_data
from modules.ml_predictor import train_ensemble_model, predict_with_model, add_technical_indicators
from modules.news_social_monitor import get_combined_sentiment
from modules.signal_aggregator import SignalAggregator
from modules.asset_selector import select_top_assets
from modules.logger import setup_logger
from modules.risk_manager import RiskManager, SignalType
from modules.optimization import optimize_autogluon

# Configure logging
logger = setup_logger(name='scheduler', level='INFO')

# Global variables
task_queue = Queue()
stop_event = threading.Event()
current_assets = []
market_data = {}
signal_aggregator = None
risk_manager = None
ml_model = None

def initialize_system():
    """Initialize the trading system components"""
    global signal_aggregator, risk_manager, current_assets
    
    logger.info("Initializing trading system")
    
    # Create required directories
    os.makedirs(SystemConfig.DEFAULT_DATA_DIR, exist_ok=True)
    os.makedirs(SystemConfig.DEFAULT_RESULTS_DIR, exist_ok=True)
    os.makedirs(SystemConfig.DEFAULT_MODELS_DIR, exist_ok=True)
    os.makedirs(SystemConfig.CACHE_DIR, exist_ok=True)
    
    # Initialize signal aggregator
    signal_aggregator = SignalAggregator(cache_dir=SystemConfig.CACHE_DIR)
    
    # Get risk parameters based on current tolerance level
    risk_params = get_risk_parameters(GlobalSettings.RISK_TOLERANCE)
    
    # Initialize risk manager
    risk_manager = RiskManager(
        max_position_size=risk_params['max_position_size'],
        max_portfolio_risk=risk_params['max_portfolio_risk'],
        stop_loss_pct=risk_params['stop_loss'],
        take_profit_pct=risk_params['take_profit'],
        position_sizing_method='volatility'
    )
    
    # Select initial assets
    if not current_assets:
        current_assets = select_top_assets(n=10)
        logger.info(f"Selected initial assets: {current_assets}")
    
    # Record system state
    system_state = {
        'timestamp': datetime.datetime.now().isoformat(),
        'trading_mode': GlobalSettings.TRADING_MODE,
        'risk_tolerance': GlobalSettings.RISK_TOLERANCE,
        'active_strategy': GlobalSettings.ACTIVE_STRATEGY,
        'current_assets': current_assets
    }
    
    state_file = os.path.join(SystemConfig.DEFAULT_DATA_DIR, 'system_state.json')
    with open(state_file, 'w') as f:
        json.dump(system_state, f, indent=4)
    
    logger.info("System initialization complete")

def update_market_data():
    """Update market data for current assets"""
    global market_data, current_assets
    
    logger.info("Updating market data")
    
    # Define date range
    end_date = datetime.datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.datetime.now() - datetime.timedelta(days=365)).strftime('%Y-%m-%d')
    
    # Fetch data for all current assets
    try:
        new_data = batch_fetch_price_data(current_assets, start_date, end_date, batch_size=5)
        
        # Update global market data
        market_data.update(new_data)
        
        # Save a snapshot of the latest data
        latest_prices = {}
        for symbol, data in new_data.items():
            if data is not None and not data.empty:
                latest_prices[symbol] = {
                    'timestamp': data.index[-1].strftime('%Y-%m-%d'),
                    'open': float(data['Open'].iloc[-1]),
                    'high': float(data['High'].iloc[-1]),
                    'low': float(data['Low'].iloc[-1]),
                    'close': float(data['Close'].iloc[-1]),
                    'volume': float(data['Volume'].iloc[-1])
                }
        
        # Save to file
        prices_file = os.path.join(SystemConfig.DEFAULT_DATA_DIR, 'latest_prices.json')
        with open(prices_file, 'w') as f:
            json.dump(latest_prices, f, indent=4)
        
        successful = sum(1 for data in new_data.values() if data is not None and not data.empty)
        logger.info(f"Updated data for {successful} out of {len(current_assets)} assets")
        
    except Exception as e:
        logger.error(f"Error updating market data: {e}")

def update_asset_universe():
    """Update the universe of assets to trade"""
    global current_assets
    
    logger.info("Updating asset universe")
    
    try:
        # Select top assets
        new_assets = select_top_assets(n=10)
        
        # Log changes
        added = [a for a in new_assets if a not in current_assets]
        removed = [a for a in current_assets if a not in new_assets]
        
        if added or removed:
            logger.info(f"Assets added: {added}")
            logger.info(f"Assets removed: {removed}")
            
            # Update current assets
            current_assets = new_assets
            
            # Save current assets
            assets_file = os.path.join(SystemConfig.DEFAULT_DATA_DIR, 'current_assets.json')
            with open(assets_file, 'w') as f:
                json.dump(current_assets, f, indent=4)
        else:
            logger.info("No changes to asset universe")
        
    except Exception as e:
        logger.error(f"Error updating asset universe: {e}")

def train_ml_models():
    """Train/update ML models for prediction"""
    global ml_model, market_data, current_assets
    
    logger.info("Training ML models")
    
    try:
        # Select a representative asset for training
        symbol = current_assets[0] if current_assets else "AAPL"
        
        # Check if we have data
        if symbol not in market_data or market_data[symbol] is None or market_data[symbol].empty:
            logger.warning(f"No data available for {symbol}, skipping ML training")
            return
        
        # Prepare data
        data = market_data[symbol].copy()
        
        # Add technical indicators
        data = add_technical_indicators(data)
        
        # Create target variable (next day return)
        data['target_return'] = data['Close'].pct_change().shift(-1)
        data.dropna(inplace=True)
        
        # Train model
        logger.info(f"Training model with {len(data)} samples")
        model = train_ensemble_model(data, label_column='target_return', time_limit=600)
        
        # Save model reference
        ml_model = model
        
        # Save model metadata
        model_metadata = {
            'timestamp': datetime.datetime.now().isoformat(),
            'symbol': symbol,
            'data_points': len(data),
            'training_time': 600,  # seconds
            'path': model.path if hasattr(model, 'path') else None
        }
        
        metadata_file = os.path.join(SystemConfig.DEFAULT_MODELS_DIR, 'model_metadata.json')
        with open(metadata_file, 'w') as f:
            json.dump(model_metadata, f, indent=4)
        
        logger.info("ML model training complete")
        
    except Exception as e:
        logger.error(f"Error training ML models: {e}")

def generate_trading_signals():
    """Generate trading signals for current assets"""
    global market_data, current_assets, signal_aggregator, risk_manager
    
    logger.info("Generating trading signals")
    
    try:
        signals = {}
        
        for symbol in current_assets:
            # Skip if no data available
            if symbol not in market_data or market_data[symbol] is None or market_data[symbol].empty:
                logger.warning(f"No data available for {symbol}, skipping signal generation")
                continue
            
            # Get price data
            price_data = market_data[symbol].copy()
            
            # Add technical indicators if not already present
            if 'RSI' not in price_data.columns:
                price_data = add_technical_indicators(price_data)
            
            # Generate signal
            signal_result = signal_aggregator.aggregate_signals(
                symbol=symbol,
                price_data=price_data,
                social_query=f"{symbol} stock",
                force_refresh=False  # Use cached signals if available
            )
            
            signals[symbol] = signal_result
            
            # Evaluate signal with risk manager
            current_price = price_data['Close'].iloc[-1]
            volatility = price_data['Close'].pct_change().std() * np.sqrt(252)
            
            risk_result = risk_manager.evaluate_signal(
                symbol=symbol,
                signal_type=SignalType.ENTRY if signal_result['signal'] > 0.5 else SignalType.EXIT,
                signal_strength=signal_result['signal'],
                price=current_price,
                volatility=volatility
            )
            
            signals[symbol]['risk_evaluation'] = risk_result
            
            logger.info(f"Signal for {symbol}: {signal_result['signal']:.2f}, Action: {risk_result['action']}")
        
        # Save signals to file
        signals_file = os.path.join(SystemConfig.DEFAULT_DATA_DIR, 'latest_signals.json')
        
        # Convert to serializable format
        serializable_signals = {}
        for symbol, signal in signals.items():
            serializable_signals[symbol] = {
                'timestamp': signal['timestamp'],
                'signal': signal['signal'],
                'action': signal['risk_evaluation']['action'] if 'risk_evaluation' in signal else 'unknown',
                'components': {k: float(v) for k, v in signal['components'].items()},
                'weights': signal['weights']
            }
        
        with open(signals_file, 'w') as f:
            json.dump(serializable_signals, f, indent=4)
        
        logger.info(f"Generated signals for {len(signals)} assets")
        
    except Exception as e:
        logger.error(f"Error generating trading signals: {e}")

def execute_paper_trades():
    """Execute paper trades based on signals"""
    logger.info("Executing paper trades")
    
    # In a real implementation, this would execute trades in a paper trading account
    # For this example, we'll just load the signals and simulate trades
    
    try:
        # Load signals
        signals_file = os.path.join(SystemConfig.DEFAULT_DATA_DIR, 'latest_signals.json')
        if not os.path.exists(signals_file):
            logger.warning("No signals file found, skipping trade execution")
            return
        
        with open(signals_file, 'r') as f:
            signals = json.load(f)
        
        # Load current portfolio
        portfolio_file = os.path.join(SystemConfig.DEFAULT_DATA_DIR, 'paper_portfolio.json')
        if os.path.exists(portfolio_file):
            with open(portfolio_file, 'r') as f:
                portfolio = json.load(f)
        else:
            # Initialize empty portfolio
            portfolio = {
                'cash': GlobalSettings.PAPER_TRADING_BALANCE,
                'positions': {},
                'trades': [],
                'equity_history': []
            }
        
        # Get latest prices
        prices_file = os.path.join(SystemConfig.DEFAULT_DATA_DIR, 'latest_prices.json')
        if not os.path.exists(prices_file):
            logger.warning("No prices file found, skipping trade execution")
            return
        
        with open(prices_file, 'r') as f:
            prices = json.load(f)
        
        # Process signals and execute trades
        for symbol, signal_data in signals.items():
            if symbol not in prices:
                continue
                
            current_price = prices[symbol]['close']
            current_action = signal_data['action']
            
            # Check if we have an existing position
            has_position = symbol in portfolio['positions']
            
            # Process actions
            if current_action == 'enter' and not has_position:
                # Calculate position size (this would normally come from risk manager)
                position_size = min(portfolio['cash'] * 0.1, 10000) / current_price
                position_size = int(position_size)  # Whole shares
                
                if position_size > 0:
                    # Execute buy
                    cost = position_size * current_price
                    portfolio['cash'] -= cost
                    
                    # Add position
                    portfolio['positions'][symbol] = {
                        'size': position_size,
                        'entry_price': current_price,
                        'entry_date': datetime.datetime.now().isoformat(),
                        'current_price': current_price,
                        'current_value': position_size * current_price
                    }
                    
                    # Record trade
                    portfolio['trades'].append({
                        'symbol': symbol,
                        'action': 'buy',
                        'size': position_size,
                        'price': current_price,
                        'value': cost,
                        'timestamp': datetime.datetime.now().isoformat()
                    })
                    
                    logger.info(f"Paper trade: BUY {position_size} {symbol} @ ${current_price:.2f}")
            
            elif current_action == 'exit' and has_position:
                # Get position details
                position = portfolio['positions'][symbol]
                position_size = position['size']
                entry_price = position['entry_price']
                
                # Calculate P&L
                pnl = position_size * (current_price - entry_price)
                pnl_pct = (current_price / entry_price - 1) * 100
                
                # Execute sell
                proceeds = position_size * current_price
                portfolio['cash'] += proceeds
                
                # Remove position
                del portfolio['positions'][symbol]
                
                # Record trade
                portfolio['trades'].append({
                    'symbol': symbol,
                    'action': 'sell',
                    'size': position_size,
                    'price': current_price,
                    'value': proceeds,
                    'pnl': pnl,
                    'pnl_pct': pnl_pct,
                    'timestamp': datetime.datetime.now().isoformat()
                })
                
                logger.info(f"Paper trade: SELL {position_size} {symbol} @ ${current_price:.2f}, P&L: ${pnl:.2f} ({pnl_pct:.2f}%)")
            
            # Update existing positions
            elif has_position:
                portfolio['positions'][symbol]['current_price'] = current_price
                portfolio['positions'][symbol]['current_value'] = portfolio['positions'][symbol]['size'] * current_price
        
        # Calculate portfolio value
        positions_value = sum(pos['current_value'] for pos in portfolio['positions'].values())
        total_value = portfolio['cash'] + positions_value
        
        # Update equity history
        portfolio['equity_history'].append({
            'timestamp': datetime.datetime.now().isoformat(),
            'cash': portfolio['cash'],
            'positions_value': positions_value,
            'total_value': total_value
        })
        
        # Save updated portfolio
        with open(portfolio_file, 'w') as f:
            json.dump(portfolio, f, indent=4)
        
        logger.info(f"Paper trading complete. Portfolio value: ${total_value:.2f}")
        
    except Exception as e:
        logger.error(f"Error executing paper trades: {e}")

def execute_live_trades():
    """Execute live trades based on signals"""
    logger.info("Live trading not implemented. This is a placeholder.")
    # This would connect to a brokerage API and execute real trades

def run_optimization():
    """Run optimization for strategy parameters"""
    logger.info("Running strategy parameter optimization")
    
    try:
        # Define parameter grid
        param_grid = {
            'stop_loss': [0.02, 0.03, 0.04],
            'take_profit': [0.04, 0.06, 0.08],
            'risk_factor': [0.005, 0.01, 0.015],
            'vol_window': [15, 20, 25],
            'rsi_period': [9, 14, 21]
        }
        
        # Run hyperparameter optimization
        best_params = optimize_autogluon(param_grid, n_trials=10)
        
        # Save results
        results_file = os.path.join(SystemConfig.DEFAULT_RESULTS_DIR, 'optimization_results.json')
        with open(results_file, 'w') as f:
            json.dump(best_params, f, indent=4)
        
        logger.info(f"Optimization complete. Best parameters: {best_params}")
        
    except Exception as e:
        logger.error(f"Error in optimization: {e}")

def generate_reports():
    """Generate performance reports"""
    logger.info("Generating performance reports")
    
    try:
        # Load portfolio data
        portfolio_file = os.path.join(SystemConfig.DEFAULT_DATA_DIR, 'paper_portfolio.json')
        if not os.path.exists(portfolio_file):
            logger.warning("No portfolio file found, skipping report generation")
            return
        
        with open(portfolio_file, 'r') as f:
            portfolio = json.load(f)
        
        # Extract equity history
        equity_history = portfolio.get('equity_history', [])
        
        if not equity_history:
            logger.warning("No equity history found, skipping report generation")
            return
        
        # Create DataFrame
        df = pd.DataFrame(equity_history)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
        
        # Calculate daily metrics
        daily_data = df.resample('D').last().dropna()
        
        # Calculate returns
        daily_data['daily_return'] = daily_data['total_value'].pct_change()
        
        # Calculate metrics
        total_return = (daily_data['total_value'].iloc[-1] / daily_data['total_value'].iloc[0] - 1) * 100
        sharpe_ratio = daily_data['daily_return'].mean() / daily_data['daily_return'].std() * np.sqrt(252)
        max_drawdown = (daily_data['total_value'] / daily_data['total_value'].cummax() - 1).min() * 100
        
        # Trade analysis
        trades = portfolio.get('trades', [])
        winning_trades = [t for t in trades if t.get('action') == 'sell' and t.get('pnl', 0) > 0]
        losing_trades = [t for t in trades if t.get('action') == 'sell' and t.get('pnl', 0) <= 0]
        
        win_rate = len(winning_trades) / max(1, len(winning_trades) + len(losing_trades)) * 100
        
        avg_win = sum(t.get('pnl', 0) for t in winning_trades) / max(1, len(winning_trades))
        avg_loss = sum(t.get('pnl', 0) for t in losing_trades) / max(1, len(losing_trades))
        
        # Create report dictionary
        report = {
            'timestamp': datetime.datetime.now().isoformat(),
            'current_value': daily_data['total_value'].iloc[-1],
            'current_cash': daily_data['cash'].iloc[-1],
            'current_positions_value': daily_data['positions_value'].iloc[-1],
            'total_return_pct': total_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown_pct': max_drawdown,
            'total_trades': len([t for t in trades if t.get('action') == 'sell']),
            'win_rate': win_rate,
            'avg_winning_trade': avg_win,
            'avg_losing_trade': avg_loss,
            'profit_factor': abs(sum(t.get('pnl', 0) for t in winning_trades) / 
                               max(0.01, abs(sum(t.get('pnl', 0) for t in losing_trades))))
        }
        
        # Save report
        report_file = os.path.join(SystemConfig.DEFAULT_REPORTS_DIR, f'performance_{datetime.datetime.now().strftime("%Y%m%d")}.json')
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=4)
        
        logger.info(f"Performance report generated: {report_file}")
        
        # Export daily data for dashboard
        daily_data.to_csv(os.path.join(SystemConfig.DEFAULT_DATA_DIR, 'trading_performance.csv'))
        
    except Exception as e:
        logger.error(f"Error generating reports: {e}")

def worker_thread():
    """Worker thread to process tasks from the queue"""
    logger.info("Worker thread started")
    
    while not stop_event.is_set():
        try:
            # Get task with a timeout (allows checking stop_event periodically)
            try:
                task = task_queue.get(timeout=1)
            except Exception:
                continue
                
            # Process task
            logger.info(f"Processing task: {task}")
            
            if task == 'initialize':
                initialize_system()
            elif task == 'update_market_data':
                update_market_data()
            elif task == 'update_asset_universe':
                update_asset_universe()
            elif task == 'train_ml_models':
                train_ml_models()
            elif task == 'generate_signals':
                generate_trading_signals()
            elif task == 'execute_paper_trades':
                execute_paper_trades()
            elif task == 'execute_live_trades':
                execute_live_trades()
            elif task == 'run_optimization':
                run_optimization()
            elif task == 'generate_reports':
                generate_reports()
            else:
                logger.warning(f"Unknown task: {task}")
            
            # Mark task as done
            task_queue.task_done()
            
        except Exception as e:
            logger.error(f"Error in worker thread: {e}")
    
    logger.info("Worker thread stopped")

def schedule_jobs():
    """Schedule recurring jobs"""
    mode = GlobalSettings.TRADING_MODE
    
    # System initialization (run immediately)
    task_queue.put('initialize')
    
    # Update market data (every 5 minutes during trading hours)
    schedule.every(5).minutes.do(lambda: task_queue.put('update_market_data'))
    
    # Update asset universe (daily)
    schedule.every().day.at("06:00").do(lambda: task_queue.put('update_asset_universe'))
    
    # Train ML models (weekly or after market close)
    schedule.every().monday.at("18:30").do(lambda: task_queue.put('train_ml_models'))
    
    # Generate trading signals (hourly or more frequently in live mode)
    if mode == 'live':
        schedule.every(15).minutes.do(lambda: task_queue.put('generate_signals'))
    else:
        schedule.every().hour.do(lambda: task_queue.put('generate_signals'))
    
    # Execute trades (based on mode)
    if mode == 'paper':
        # Paper trading (hourly)
        schedule.every().hour.do(lambda: task_queue.put('execute_paper_trades'))
    elif mode == 'live':
        # Live trading (after signals generation)
        schedule.every(16).minutes.do(lambda: task_queue.put('execute_live_trades'))
    
    # Run optimization (weekly)
    schedule.every().saturday.at("10:00").do(lambda: task_queue.put('run_optimization'))
    
    # Generate reports (daily after market close)
    schedule.every().day.at("17:30").do(lambda: task_queue.put('generate_reports'))
    
    logger.info("Jobs scheduled successfully")

def signal_handler(signum, frame):
    """Handle termination signals"""
    logger.info(f"Received signal {signum}, shutting down...")
    stop_event.set()

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Trading System Scheduler')
    
    parser.add_argument('--mode', choices=['backtest', 'paper', 'live'], default=GlobalSettings.TRADING_MODE,
                      help='Trading mode')
    
    parser.add_argument('--risk', choices=['low', 'medium', 'high'], default=GlobalSettings.RISK_TOLERANCE,
                      help='Risk tolerance level')
    
    parser.add_argument('--dashboard', action='store_true',
                      help='Launch dashboard')
    
    return parser.parse_args()

def main():
    """Main entry point"""
    # Parse arguments
    args = parse_arguments()
    
    # Update global settings
    GlobalSettings.TRADING_MODE = args.mode
    GlobalSettings.RISK_TOLERANCE = args.risk
    
    logger.info(f"Starting scheduler in {args.mode} mode with {args.risk} risk tolerance")
    
    # Launch dashboard if requested
    if args.dashboard:
        import subprocess
        import sys
        
        dashboard_path = os.path.join('dashboards', 'dashboard.py')
        subprocess.Popen([
            sys.executable, "-m", "streamlit", "run", 
            dashboard_path, "--server.port", str(SystemConfig.DASHBOARD_PORT)
        ])
        logger.info(f"Dashboard launched at: http://localhost:{SystemConfig.DASHBOARD_PORT}")
    
    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Start worker thread
    worker = threading.Thread(target=worker_thread)
    worker.daemon = True
    worker.start()
    
    # Schedule jobs
    schedule_jobs()
    
    # Main loop
    logger.info("Entering main loop")
    try:
        while not stop_event.is_set():
            schedule.run_pending()
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received, shutting down...")
        stop_event.set()
    
    # Wait for worker to finish
    worker.join(timeout=5)
    logger.info("Scheduler stopped")

if __name__ == "__main__":
    main()