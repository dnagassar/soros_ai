# Add these imports at the top
from modules.broker_integration import BrokerIntegration
from modules.ml_pipeline import MLPipeline
import numpy as np

# Add these to global variables section
broker = None
ml_pipeline = None

# Update initialize_system function
def initialize_system():
    """Initialize the trading system components"""
    global signal_aggregator, risk_manager, current_assets, broker, ml_pipeline
    
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
    
    # Initialize broker integration for paper or live trading
    if GlobalSettings.TRADING_MODE in ['paper', 'live']:
        broker = BrokerIntegration(paper=GlobalSettings.TRADING_MODE == 'paper')
        logger.info(f"Broker integration initialized in {GlobalSettings.TRADING_MODE} mode")
    
    # Initialize ML pipeline
    ml_pipeline = MLPipeline(model_dir=SystemConfig.DEFAULT_MODELS_DIR)
    
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
        'current_assets': current_assets,
        'broker_connected': broker.check_connection() if broker else False
    }
    
    state_file = os.path.join(SystemConfig.DEFAULT_DATA_DIR, 'system_state.json')
    with open(state_file, 'w') as f:
        json.dump(system_state, f, indent=4)
    
    logger.info("System initialization complete")

# Update train_ml_models function
def train_ml_models():
    """Train/update ML models for prediction"""
    global ml_pipeline, market_data, current_assets
    
    logger.info("Training ML models")
    
    try:
        # Process each asset
        for symbol in current_assets:
            # Skip if no data available
            if symbol not in market_data or market_data[symbol] is None or market_data[symbol].empty:
                logger.warning(f"No data available for {symbol}, skipping ML training")
                continue
            
            # Prepare data
            price_data = market_data[symbol].copy()
            
            # Prepare features
            X, y, feature_cols = ml_pipeline.prepare_features(price_data, symbol)
            
            if X is not None and y is not None:
                # Train model
                logger.info(f"Training ML model for {symbol} with {len(X)} samples")
                
                # Use 80% of data for training
                train_size = int(len(X) * 0.8)
                X_train, y_train = X.iloc[:train_size], y.iloc[:train_size]
                
                # Train model
                predictor, performance = ml_pipeline.train_model(X_train, y_train, symbol)
                
                if predictor is not None:
                    # Save model
                    ml_pipeline.save_model(symbol)
                    
                    logger.info(f"ML model for {symbol} trained successfully")
                    if performance:
                        logger.info(f"Model performance: {performance}")
            else:
                logger.warning(f"Feature preparation failed for {symbol}")
        
    except Exception as e:
        logger.error(f"Error training ML models: {e}")

# Replace execute_paper_trades function
def execute_paper_trades():
    """Execute paper trades based on signals"""
    global broker, risk_manager
    
    logger.info("Executing paper trades")
    
    # If broker is not initialized, use internal simulation
    if broker is None:
        broker = BrokerIntegration(paper=True)
        logger.info("Initialized broker for paper trading")
    
    # Use broker integration for paper trading
    try:
        # Load signals
        signals_file = os.path.join(SystemConfig.DEFAULT_DATA_DIR, 'latest_signals.json')
        if not os.path.exists(signals_file):
            logger.warning("No signals file found, skipping trade execution")
            return
        
        with open(signals_file, 'r') as f:
            signals = json.load(f)
        
        # Execute trades
        result = broker.execute_trades(signals, risk_manager)
        
        if result.get('success', False):
            logger.info(f"Paper trades executed: {len(result.get('orders', []))} orders placed")
        else:
            logger.warning(f"Paper trade execution failed: {result.get('message', 'Unknown error')}")
            
    except Exception as e:
        logger.error(f"Error executing paper trades: {e}")

# Replace execute_live_trades function
def execute_live_trades():
    """Execute live trades based on signals"""
    global broker, risk_manager
    
    logger.info("Executing live trades")
    
    # Verify that we're in live mode
    if GlobalSettings.TRADING_MODE != 'live':
        logger.warning("Not in live trading mode, skipping live trade execution")
        return
    
    # Verify broker connection
    if broker is None:
        broker = BrokerIntegration(paper=False)
        logger.info("Initialized broker for live trading")
    
    # Execute trades
    try:
        # Load signals
        signals_file = os.path.join(SystemConfig.DEFAULT_DATA_DIR, 'latest_signals.json')
        if not os.path.exists(signals_file):
            logger.warning("No signals file found, skipping trade execution")
            return
        
        with open(signals_file, 'r') as f:
            signals = json.load(f)
        
        # Execute trades
        result = broker.execute_trades(signals, risk_manager)
        
        if result.get('success', False):
            logger.info(f"Live trades executed: {len(result.get('orders', []))} orders placed")
        else:
            logger.warning(f"Live trade execution failed: {result.get('message', 'Unknown error')}")
            
    except Exception as e:
        logger.error(f"Error executing live trades: {e}")