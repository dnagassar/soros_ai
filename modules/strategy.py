# modules/strategy.py
import backtrader as bt
import numpy as np
import pandas as pd
import logging
from datetime import datetime, timedelta
from modules.sentiment_analysis import aggregate_sentiments
from modules.news_social_monitor import get_combined_sentiment
from modules.risk_manager import RiskManager, SignalType

# Configure logging
logger = logging.getLogger(__name__)

class AdaptiveSentimentStrategy(bt.Strategy):
    """
    Enhanced strategy that combines technical indicators, sentiment analysis,
    and machine learning signals with adaptive position sizing and risk management.
    """
    
    params = (
        ('sentiment_period', 3),      # Days to look back for sentiment
        ('vol_window', 20),           # Window for volatility calculation
        ('atr_period', 14),           # ATR period
        ('ema_short', 10),            # Short EMA period
        ('ema_medium', 30),           # Medium EMA period
        ('ema_long', 50),             # Long EMA period
        ('rsi_period', 14),           # RSI period
        ('rsi_overbought', 70),       # RSI overbought threshold
        ('rsi_oversold', 30),         # RSI oversold threshold
        ('macd_fast', 12),            # MACD fast period
        ('macd_slow', 26),            # MACD slow period
        ('macd_signal', 9),           # MACD signal period
        ('stop_loss', 0.03),          # Stop loss percentage
        ('take_profit', 0.05),        # Take profit percentage
        ('trailing_stop', False),     # Use trailing stop
        ('trailing_percent', 0.02),   # Trailing stop percentage
        ('risk_factor', 0.01),        # Risk per trade as fraction of portfolio
        ('use_ml_signals', True),     # Whether to use ML signals
        ('ml_weight', 0.4),           # Weight of ML signals in decision
        ('sentiment_weight', 0.3),    # Weight of sentiment in decision
        ('tech_weight', 0.3),         # Weight of technical indicators in decision
        ('risk_manager', None),       # External risk manager (optional)
        ('social_query', None),       # Social media query string (optional)
        ('rebalance_freq', 5),        # Rebalance frequency in days
        ('entry_threshold', 0.5),     # Signal threshold for entry
        ('exit_threshold', -0.3),     # Signal threshold for exit
    )
    
    def __init__(self):
        """Initialize the strategy with indicators and logging"""
        self.orders = {}  # Tracking open orders
        self.positions_info = {}  # Detailed position tracking
        self.last_sentiment_check = None
        self.sentiments = {}
        self.ml_signals = {}
        self.last_rebalance = None
        
        # Initialize risk manager if not provided
        if self.p.risk_manager is None:
            self.risk_manager = RiskManager(
                stop_loss_pct=self.p.stop_loss,
                take_profit_pct=self.p.take_profit,
                position_sizing_method='volatility'
            )
        else:
            self.risk_manager = self.p.risk_manager
        
        # Dictionary to store indicators for each data feed
        self.indicators = {}
        
        # Create indicators for each data feed
        for i, data in enumerate(self.datas):
            self.indicators[data] = {
                # Moving Averages
                'ema_short': bt.indicators.ExponentialMovingAverage(data.close, period=self.p.ema_short),
                'ema_medium': bt.indicators.ExponentialMovingAverage(data.close, period=self.p.ema_medium),
                'ema_long': bt.indicators.ExponentialMovingAverage(data.close, period=self.p.ema_long),
                
                # RSI
                'rsi': bt.indicators.RelativeStrengthIndex(data.close, period=self.p.rsi_period),
                
                # MACD
                'macd': bt.indicators.MACD(
                    data.close, 
                    period_me1=self.p.macd_fast,
                    period_me2=self.p.macd_slow, 
                    period_signal=self.p.macd_signal
                ),
                
                # Bollinger Bands
                'bbands': bt.indicators.BollingerBands(data.close, period=20),
                
                # Average True Range for volatility
                'atr': bt.indicators.ATR(data, period=self.p.atr_period),
                
                # Volume indicators
                'volume_sma': bt.indicators.SMA(data.volume, period=20),
                
                # Volatility
                'volatility': bt.indicators.StdDev(data.close, period=self.p.vol_window),
            }
            
            # Track crossovers for signals
            self.indicators[data]['ema_cross'] = bt.indicators.CrossOver(
                self.indicators[data]['ema_short'],
                self.indicators[data]['ema_medium']
            )
            
            self.indicators[data]['macd_cross'] = bt.indicators.CrossOver(
                self.indicators[data]['macd'].macd,
                self.indicators[data]['macd'].signal
            )
        
        logger.info(f"Strategy initialized with {len(self.datas)} data feeds")
    
    def start(self):
        """Called when strategy is started"""
        self.log('Strategy started')
        
        # Initialize portfolio value in risk manager
        self.risk_manager.update_portfolio_value(self.broker.getvalue())
        
        # Initialize last rebalance date
        self.last_rebalance = self.datetime.date()
    
    def log(self, txt, dt=None):
        """Logging function"""
        dt = dt or self.datetime.date()
        logger.info(f'{dt.isoformat()} - {txt}')
    
    def notify_order(self, order):
        """Handle order notifications"""
        if order.status in [order.Submitted, order.Accepted]:
            # Order is submitted/accepted - no action needed
            return
        
        # Get the data for this order
        data = order.data
        symbol = data._name
        
        # Check if an order has been completed
        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(f'BUY EXECUTED for {symbol}, Price: {order.executed.price:.2f}, Size: {order.executed.size:.0f}')
                
                # Update position info
                if symbol not in self.positions_info:
                    self.positions_info[symbol] = {
                        'entry_price': order.executed.price,
                        'size': order.executed.size,
                        'entry_date': self.datetime.date(),
                        'stop_price': order.executed.price * (1 - self.p.stop_loss),
                        'take_profit': order.executed.price * (1 + self.p.take_profit)
                    }
                else:
                    # Update existing position
                    avg_price = (self.positions_info[symbol]['entry_price'] * self.positions_info[symbol]['size'] + 
                                order.executed.price * order.executed.size) / (self.positions_info[symbol]['size'] + order.executed.size)
                    self.positions_info[symbol]['entry_price'] = avg_price
                    self.positions_info[symbol]['size'] += order.executed.size
                    self.positions_info[symbol]['stop_price'] = avg_price * (1 - self.p.stop_loss)
                    self.positions_info[symbol]['take_profit'] = avg_price * (1 + self.p.take_profit)
                
                # Update position in risk manager
                self.risk_manager.update_position(
                    symbol=symbol,
                    current_price=order.executed.price,
                    size=self.positions_info[symbol]['size'],
                    cost_basis=self.positions_info[symbol]['entry_price'],
                    timestamp=datetime.combine(self.datetime.date(), datetime.min.time())
                )
                
            elif order.issell():
                self.log(f'SELL EXECUTED for {symbol}, Price: {order.executed.price:.2f}, Size: {order.executed.size:.0f}')
                
                # If closing a position
                if symbol in self.positions_info and order.executed.size >= self.positions_info[symbol]['size']:
                    # Calculate P&L
                    entry_price = self.positions_info[symbol]['entry_price']
                    pnl = (order.executed.price - entry_price) * order.executed.size
                    pnl_pct = (order.executed.price / entry_price - 1) * 100
                    
                    self.log(f'P&L for {symbol}: ${pnl:.2f} ({pnl_pct:.2f}%)')
                    
                    # Close position in risk manager
                    self.risk_manager.close_position(
                        symbol=symbol,
                        exit_price=order.executed.price,
                        timestamp=datetime.combine(self.datetime.date(), datetime.min.time()),
                        reason="Strategy signal"
                    )
                    
                    # Remove from positions_info
                    del self.positions_info[symbol]
                
                # If reducing a position
                elif symbol in self.positions_info:
                    self.positions_info[symbol]['size'] -= order.executed.size
                    
                    # Update position in risk manager
                    self.risk_manager.update_position(
                        symbol=symbol,
                        current_price=order.executed.price,
                        size=self.positions_info[symbol]['size'],
                        timestamp=datetime.combine(self.datetime.date(), datetime.min.time())
                    )
            
            # Remove order from tracking
            if symbol in self.orders:
                del self.orders[symbol]
        
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log(f'Order for {symbol} {order.Status[order.status]}')
            
            # Remove order from tracking
            if symbol in self.orders:
                del self.orders[symbol]
    
    def notify_trade(self, trade):
        """Handle trade notifications"""
        if not trade.isclosed:
            return
        
        # Get the data for this trade
        data = trade.data
        symbol = data._name
        
        self.log(f'TRADE for {symbol} - Profit: ${trade.pnl:.2f}, Profit %: {(trade.pnl/trade.price)*100:.2f}%')
    
    def get_sentiment(self, data):
        """Get sentiment for a symbol from external data sources"""
        symbol = data._name
        current_date = self.datetime.date()
        
        # Only update sentiment periodically to avoid excessive API calls
        if (symbol not in self.sentiments or 
            self.last_sentiment_check is None or 
            (current_date - self.last_sentiment_check).days >= self.p.sentiment_period):
            
            try:
                # Generate text for sentiment analysis
                news_text = f"Latest market news for {symbol}"
                
                # Get sentiment from news
                base_sentiment = aggregate_sentiments(news_text)
                
                # Get social media sentiment if enabled
                if self.p.social_query:
                    query = self.p.social_query.replace('{symbol}', symbol)
                    social_sentiment = get_combined_sentiment(query, symbol=symbol)
                    
                    # Combine sentiments (weighted)
                    combined_score = base_sentiment.get('score', 0) * 0.6 + social_sentiment.get('score', 0) * 0.4
                else:
                    combined_score = base_sentiment.get('score', 0)
                
                # Normalize to -1 to 1 range
                normalized_score = max(min(combined_score, 1), -1)
                
                # Store sentiment
                self.sentiments[symbol] = normalized_score
                self.last_sentiment_check = current_date
                
                self.log(f'Updated sentiment for {symbol}: {normalized_score:.2f}')
                
            except Exception as e:
                logger.error(f"Error getting sentiment for {symbol}: {e}")
                self.sentiments[symbol] = 0  # Use neutral sentiment on error
        
        return self.sentiments.get(symbol, 0)
    
    def get_ml_signal(self, data):
        """Get ML prediction signal for a symbol (placeholder)"""
        symbol = data._name
        
        # In a real implementation, this would call your ML model
        if self.p.use_ml_signals:
            try:
                # For demonstration, generate a random signal biased by trend
                trend = self.get_trend_signal(data)
                random_component = np.random.normal(0, 0.2)
                ml_signal = 0.7 * trend + 0.3 * random_component
                
                # Clamp to -1 to 1 range
                ml_signal = max(min(ml_signal, 1), -1)
                
                # Store the signal
                self.ml_signals[symbol] = ml_signal
                
            except Exception as e:
                logger.error(f"Error getting ML signal for {symbol}: {e}")
                self.ml_signals[symbol] = 0  # Neutral on error
        else:
            self.ml_signals[symbol] = 0
        
        return self.ml_signals.get(symbol, 0)
    
    def get_technical_signal(self, data):
        """Calculate technical indicator signals"""
        ind = self.indicators[data]
        
        signals = []
        
        # EMA Trend signal (-1 to 1)
        if ind['ema_short'][-1] > ind['ema_long'][-1]:
            ema_signal = min((ind['ema_short'][-1] / ind['ema_long'][-1] - 1) * 5, 1)
        else:
            ema_signal = max((ind['ema_short'][-1] / ind['ema_long'][-1] - 1) * 5, -1)
        signals.append(ema_signal)
        
        # RSI signal (-1 to 1)
        rsi_val = ind['rsi'][-1]
        if rsi_val > self.p.rsi_overbought:
            rsi_signal = -((rsi_val - self.p.rsi_overbought) / (100 - self.p.rsi_overbought))
        elif rsi_val < self.p.rsi_oversold:
            rsi_signal = (self.p.rsi_oversold - rsi_val) / self.p.rsi_oversold
        else:
            rsi_signal = (2 * (rsi_val - self.p.rsi_oversold) / (self.p.rsi_overbought - self.p.rsi_oversold)) - 1
        signals.append(rsi_signal)
        
        # MACD signal (-1 to 1)
        macd_val = ind['macd'].macd[-1]
        signal_val = ind['macd'].signal[-1]
        if abs(signal_val) > 0:
            macd_signal = min(max((macd_val - signal_val) / abs(signal_val) * 2, -1), 1)
        else:
            macd_signal = 0
        signals.append(macd_signal)
        
        # Bollinger Band signal (-1 to 1)
        bb = ind['bbands']
        price = data.close[-1]
        bb_signal = 0
        
        if price > bb.top[-1]:
            bb_signal = -1  # Overbought
        elif price < bb.bot[-1]:
            bb_signal = 1   # Oversold
        else:
            # Normalize position within the bands
            band_width = bb.top[-1] - bb.bot[-1]
            if band_width > 0:
                rel_position = (price - bb.mid[-1]) / (band_width / 2)
                bb_signal = -rel_position  # Higher in band = more negative
            
        signals.append(bb_signal)
        
        # Volume signal (-1 to 1)
        volume = data.volume[-1]
        avg_volume = ind['volume_sma'][-1]
        if avg_volume > 0:
            vol_signal = min(max((volume / avg_volume - 1) * 2, -1), 1)
        else:
            vol_signal = 0
        signals.append(vol_signal * 0.5)  # Reduce volume impact
        
        # Crossover signals (stronger but shorter-lived signals)
        if ind['ema_cross'][-1] == 1:  # Bullish crossover
            signals.append(1)
        elif ind['ema_cross'][-1] == -1:  # Bearish crossover
            signals.append(-1)
        
        if ind['macd_cross'][-1] == 1:  # Bullish MACD crossover
            signals.append(1)
        elif ind['macd_cross'][-1] == -1:  # Bearish MACD crossover
            signals.append(-1)
        
        # Calculate volatility-adjusted average signal
        if signals:
            # Calculate the volatility measure (0-1 scale)
            volatility = min(ind['volatility'][-1] / data.close[-1], 0.1) * 10  # Scale to 0-1
            
            # Adjust signal strength by volatility (reduce signal in high volatility)
            volatility_factor = 1 - volatility * 0.5
            
            # Weighted average of signals
            tech_signal = sum(signals) / len(signals) * volatility_factor
            return max(min(tech_signal, 1), -1)  # Ensure -1 to 1 range
        
        return 0
    
    def get_trend_signal(self, data):
        """Get trend signal from moving averages"""
        ind = self.indicators[data]
        
        # Check short-term trend (current vs short-term past)
        short_trend = data.close[-1] / data.close[-5] - 1 if len(data) >= 5 else 0
        
        # Check medium-term trend using EMAs
        medium_trend = ind['ema_short'][-1] / ind['ema_medium'][-1] - 1
        
        # Check long-term trend
        long_trend = ind['ema_medium'][-1] / ind['ema_long'][-1] - 1
        
        # Weight the trends
        trend_signal = short_trend * 0.5 + medium_trend * 0.3 + long_trend * 0.2
        
        # Scale to a -1 to 1 range (assuming trends usually within ±10%)
        trend_signal = max(min(trend_signal * 10, 1), -1)
        
        return trend_signal
    
    def should_rebalance(self):
        """Check if it's time to rebalance the portfolio"""
        if self.last_rebalance is None:
            return True
        
        current_date = self.datetime.date()
        days_since_rebalance = (current_date - self.last_rebalance).days
        
        return days_since_rebalance >= self.p.rebalance_freq
    
    def next(self):
        """Main strategy method called on each bar"""
        # Skip the first few bars until we have enough data for indicators
        min_period = max(self.p.ema_long, self.p.vol_window, self.p.atr_period)
        if len(self.data) < min_period:
            return
        
        # Update portfolio value in risk manager
        self.risk_manager.update_portfolio_value(
            self.broker.getvalue(),
            timestamp=datetime.combine(self.datetime.date(), datetime.min.time())
        )
        
        # Update market state in risk manager
        avg_volatility = 0
        for data in self.datas:
            if data._name in self.indicators:
                vol = self.indicators[data]['volatility'][-1] / data.close[-1]
                avg_volatility += vol
        
        if len(self.datas) > 0:
            avg_volatility /= len(self.datas)
        
        self.risk_manager.update_market_state(
            volatility=min(avg_volatility * 10, 1),  # Scale to 0-1
            trend=self.get_trend_signal(self.data0)  # Use first data feed as proxy
        )
        
        # Process each data feed
        for i, data in enumerate(self.datas):
            self.process_symbol(data)
        
        # Check if rebalancing is needed
        if self.should_rebalance():
            self.rebalance_portfolio()
    
    def process_symbol(self, data):
        """Process a single symbol"""
        symbol = data._name
        
        # Skip if we have an open order
        if symbol in self.orders:
            return
        
        # Get current position
        position = self.getposition(data)
        
        # Update position in risk manager with current price
        if position.size != 0 and symbol in self.positions_info:
            self.risk_manager.update_position(
                symbol=symbol,
                current_price=data.close[0],
                size=position.size,
                timestamp=datetime.combine(self.datetime.date(), datetime.min.time())
            )
        
        # Check for stop loss or take profit for existing positions
        if position.size > 0 and symbol in self.positions_info:
            # Check stop loss
            if self.risk_manager.check_stop_loss(symbol, data.close[0]):
                self.log(f'STOP LOSS triggered for {symbol} at {data.close[0]:.2f}')
                self.sell(data=data, size=position.size)
                self.orders[symbol] = 'sell'
                return
            
            # Check take profit
            if self.risk_manager.check_take_profit(symbol, data.close[0]):
                self.log(f'TAKE PROFIT triggered for {symbol} at {data.close[0]:.2f}')
                self.sell(data=data, size=position.size)
                self.orders[symbol] = 'sell'
                return
            
            # Update trailing stop if enabled
            if self.p.trailing_stop:
                trailing_pct = self.p.trailing_percent
                self.risk_manager.adjust_stop_loss(symbol, trailing_pct=trailing_pct)
        
        # Calculate combined signal
        tech_signal = self.get_technical_signal(data)
        sentiment_signal = self.get_sentiment(data)
        ml_signal = self.get_ml_signal(data)
        
        # Weighted combination
        combined_signal = (
            tech_signal * self.p.tech_weight +
            sentiment_signal * self.p.sentiment_weight +
            ml_signal * self.p.ml_weight
        )
        
        # Log signals (less frequently to avoid log spam)
        if len(self.data0) % 5 == 0:
            self.log(f'Signals for {symbol}: Tech={tech_signal:.2f}, Sentiment={sentiment_signal:.2f}, ML={ml_signal:.2f}, Combined={combined_signal:.2f}')
        
        # Evaluate signal with risk manager
        if position.size == 0:  # No position
            if combined_signal > self.p.entry_threshold:
                result = self.risk_manager.evaluate_signal(
                    symbol=symbol,
                    signal_type=SignalType.ENTRY,
                    signal_strength=combined_signal,
                    price=data.close[0],
                    volatility=self.indicators[data]['volatility'][-1] / data.close[0],
                    win_probability=0.5 + combined_signal * 0.2,  # Estimate based on signal strength
                    expected_return=self.p.take_profit
                )
                
                if result['action'] == 'enter':
                    self.log(f'BUY SIGNAL for {symbol} - Signal: {combined_signal:.2f}, Size: {result["size"]}')
                    self.buy(data=data, size=result['size'])
                    self.orders[symbol] = 'buy'
        
        elif position.size > 0:  # Existing long position
            if combined_signal < self.p.exit_threshold:
                result = self.risk_manager.evaluate_signal(
                    symbol=symbol,
                    signal_type=SignalType.EXIT,
                    signal_strength=combined_signal,
                    price=data.close[0]
                )
                
                if result['action'] == 'exit':
                    self.log(f'SELL SIGNAL for {symbol} - Signal: {combined_signal:.2f}, Size: {position.size}')
                    self.sell(data=data, size=position.size)
                    self.orders[symbol] = 'sell'
    
    def rebalance_portfolio(self):
        """Rebalance the portfolio based on current signals"""
        self.log('Rebalancing portfolio')
        self.last_rebalance = self.datetime.date()
        
        # Get current portfolio stats
        portfolio_stats = self.risk_manager.get_portfolio_stats()
        
        # Calculate target allocation for each asset
        total_signal = 0
        signals = {}
        
        for data in self.datas:
            symbol = data._name
            
            # Skip if we have an open order
            if symbol in self.orders:
                continue
            
            # Calculate combined signal
            tech_signal = self.get_technical_signal(data)
            sentiment_signal = self.get_sentiment(data)
            ml_signal = self.get_ml_signal(data)
            
            combined_signal = (
                tech_signal * self.p.tech_weight +
                sentiment_signal * self.p.sentiment_weight +
                ml_signal * self.p.ml_weight
            )
            
            # Only consider positive signals for allocation
            if combined_signal > 0:
                signals[symbol] = combined_signal
                total_signal += combined_signal
        
        if total_signal > 0:
            # Calculate target allocations based on signal strength
            target_allocations = {}
            for symbol, signal in signals.items():
                target_allocations[symbol] = signal / total_signal
                
                # Apply max position size limit
                target_allocations[symbol] = min(target_allocations[symbol], self.p.risk_factor * 5)
            
            # Normalize allocations if needed
            total_allocation = sum(target_allocations.values())
            if total_allocation > 0:
                for symbol in target_allocations:
                    target_allocations[symbol] /= total_allocation
            
            # Calculate target position sizes
            for data in self.datas:
                symbol = data._name
                
                # Skip if we have an open order
                if symbol in self.orders:
                    continue
                
                current_position = self.getposition(data)
                
                # Target allocation
                target_alloc = target_allocations.get(symbol, 0)
                
                if target_alloc > 0:
                    # Calculate target value
                    target_value = self.broker.getvalue() * target_alloc
                    
                    # Calculate target size
                    target_size = int(target_value / data.close[0])
                    
                    # Current allocation
                    current_value = current_position.size * data.close[0]
                    current_alloc = current_value / self.broker.getvalue() if self.broker.getvalue() > 0 else 0
                    
                    # Check if rebalancing is needed (avoid small adjustments)
                    if abs(current_alloc - target_alloc) > self.p.rebalance_threshold / 2:
                        if target_size > current_position.size:
                            # Buy more
                            size_diff = target_size - current_position.size
                            if size_diff > 0:
                                self.log(f'REBALANCE BUY for {symbol} - Target: {target_alloc:.2%}, Current: {current_alloc:.2%}, Size: {size_diff}')
                                self.buy(data=data, size=size_diff)
                                self.orders[symbol] = 'buy'
                        
                        elif target_size < current_position.size:
                            # Reduce position
                            size_diff = current_position.size - target_size
                            if size_diff > 0:
                                self.log(f'REBALANCE SELL for {symbol} - Target: {target_alloc:.2%}, Current: {current_alloc:.2%}, Size: {size_diff}')
                                self.sell(data=data, size=size_diff)
                                self.orders[symbol] = 'sell'
                
                elif current_position.size > 0:
                    # Exit position if target allocation is zero
                    self.log(f'REBALANCE EXIT for {symbol} - Target: 0%, Current: {current_alloc:.2%}, Size: {current_position.size}')
                    self.sell(data=data, size=current_position.size)
                    self.orders[symbol] = 'sell'
    
    def stop(self):
        """Called when the strategy is stopped"""
        self.log('Strategy stopped')
        
        # Calculate final returns
        starting_value = self.broker.startingcash
        final_value = self.broker.getvalue()
        returns_pct = (final_value / starting_value - 1) * 100
        
        self.log(f'Starting Value: ${starting_value:.2f}')
        self.log(f'Final Value: ${final_value:.2f}')
        self.log(f'Returns: {returns_pct:.2f}%')
        
        # Log trade stats
        trade_stats = self.risk_manager.get_portfolio_stats()
        self.log(f'Win Rate: {trade_stats.get("win_rate", 0):.2f}%')
        self.log(f'Total Trades: {trade_stats.get("total_trades", 0)}')
        self.log(f'Total P&L: ${trade_stats.get("total_pnl", 0):.2f}')


# Additional strategy variations can be implemented here

class MACDStrategy(bt.Strategy):
    """Simple MACD strategy for comparison purposes"""
    
    params = (
        ('macd_fast', 12),
        ('macd_slow', 26),
        ('macd_signal', 9),
        ('stop_loss', 0.03),
    )
    
    def __init__(self):
        self.macd = {}
        self.order = None
        
        for i, data in enumerate(self.datas):
            self.macd[data] = bt.indicators.MACD(
                data,
                period_me1=self.p.macd_fast,
                period_me2=self.p.macd_slow,
                period_signal=self.p.macd_signal
            )
    
    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            return
            
        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(f'BUY EXECUTED - Price: {order.executed.price:.2f}, Size: {order.executed.size:.0f}')
            elif order.issell():
                self.log(f'SELL EXECUTED - Price: {order.executed.price:.2f}, Size: {order.executed.size:.0f}')
            
        self.order = None
    
    def log(self, txt, dt=None):
        dt = dt or self.datetime.date()
        logger.info(f'{dt.isoformat()} - {txt}')
    
    def next(self):
        if self.order:
            return
            
        for data in self.datas:
            position = self.getposition(data)
            
            # No position - check for buy signal
            if not position:
                if self.macd[data].macd[0] > self.macd[data].signal[0] and self.macd[data].macd[-1] <= self.macd[data].signal[-1]:
                    size = int(self.broker.getcash() * 0.95 / data.close[0])
                    if size > 0:
                        self.log(f'BUY SIGNAL - Price: {data.close[0]:.2f}, Size: {size}')
                        self.order = self.buy(data=data, size=size)
            
            # Have position - check for sell signal
            else:
                # Check MACD crossover (sell signal)
                if self.macd[data].macd[0] < self.macd[data].signal[0] and self.macd[data].macd[-1] >= self.macd[data].signal[-1]:
                    self.log(f'SELL SIGNAL - Price: {data.close[0]:.2f}, Size: {position.size}')
                    self.order = self.sell(data=data, size=position.size)
                
                # Check stop loss
                elif data.close[0] < position.price * (1 - self.p.stop_loss):
                    self.log(f'STOP LOSS - Price: {data.close[0]:.2f}, Size: {position.size}')
                    self.order = self.sell(data=data, size=position.size)


if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    logger.info("Strategy module initialized")