# modules/risk_manager.py
import numpy as np
import pandas as pd
import logging
import datetime
from enum import Enum

logger = logging.getLogger(__name__)

class SignalType(Enum):
    """Enum for signal types"""
    ENTRY = 1
    EXIT = 2
    ADJUST = 3

class RiskLevel(Enum):
    """Enum for market risk levels"""
    LOW = 1      # Normal market conditions
    MEDIUM = 2   # Elevated volatility
    HIGH = 3     # Extreme volatility/crisis
    EXTREME = 4  # Market emergency

class RiskManager:
    """
    Comprehensive risk management system that implements:
    - Position sizing based on volatility
    - Dynamic stop loss and take profit levels
    - Portfolio-level risk controls
    - Circuit breakers for unusual market conditions
    - Kelly criterion for optimal position sizing
    """
    
    def __init__(self, 
                max_position_size=0.1,  # Maximum size for any single position (% of portfolio)
                max_portfolio_risk=0.02,  # Maximum portfolio risk per day (% of portfolio)
                min_position_size=0.01,  # Minimum position size (% of portfolio)
                stop_loss_pct=0.02,  # Default stop loss percentage
                take_profit_pct=0.04,  # Default take profit percentage 
                max_drawdown_limit=0.1,  # Circuit breaker for max drawdown (% of portfolio)
                volatility_lookback=20,  # Number of days to use for volatility calculation
                kelly_fraction=0.3,  # Conservative fraction of Kelly criterion to use
                position_sizing_method='equal',  # 'equal', 'volatility', 'kelly'
                rebalance_threshold=0.05):  # Threshold for position rebalancing (% deviation)
        
        self.max_position_size = max_position_size
        self.min_position_size = min_position_size
        self.max_portfolio_risk = max_portfolio_risk
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.max_drawdown_limit = max_drawdown_limit
        self.volatility_lookback = volatility_lookback
        self.kelly_fraction = kelly_fraction
        self.position_sizing_method = position_sizing_method
        self.rebalance_threshold = rebalance_threshold
        
        # Market state tracking
        self.risk_level = RiskLevel.LOW
        self.market_state = {
            'volatility': 0.0,
            'trend': 0.0,
            'correlation': 0.0,
            'liquidity': 1.0,
            'sentiment': 0.0,
        }
        
        # Portfolio state tracking
        self.portfolio_value = 0.0
        self.positions = {}  # Symbol -> position details
        self.portfolio_history = []  # List of historical portfolio values
        self.max_portfolio_value = 0.0  # For drawdown calculation
        self.current_drawdown = 0.0
        
        # Performance tracking
        self.trade_history = []
        self.win_count = 0
        self.loss_count = 0
        self.total_pnl = 0.0
        
        logger.info(f"Risk Manager initialized with position sizing method: {position_sizing_method}")
    
    def update_portfolio_value(self, new_value, timestamp=None):
        """
        Update the current portfolio value and track history
        
        Parameters:
          - new_value: Current portfolio value
          - timestamp: Optional timestamp (defaults to now)
        """
        timestamp = timestamp or datetime.datetime.now()
        
        self.portfolio_value = new_value
        self.portfolio_history.append({
            'timestamp': timestamp,
            'value': new_value
        })
        
        # Update maximum portfolio value
        if new_value > self.max_portfolio_value:
            self.max_portfolio_value = new_value
        
        # Calculate current drawdown
        if self.max_portfolio_value > 0:
            self.current_drawdown = 1 - (new_value / self.max_portfolio_value)
        
        # Check for circuit breaker conditions
        self._check_circuit_breakers()
        
        logger.debug(f"Portfolio value updated: ${new_value:.2f}, Drawdown: {self.current_drawdown:.2%}")
    
    def update_market_state(self, 
                           volatility=None, 
                           trend=None, 
                           correlation=None, 
                           liquidity=None, 
                           sentiment=None):
        """
        Update the current market state parameters
        
        Parameters:
          - volatility: Current market volatility measure (0-1)
          - trend: Market trend strength (-1 to 1)
          - correlation: Average correlation between assets (0-1)
          - liquidity: Liquidity measure (0-1)
          - sentiment: Market sentiment measure (-1 to 1)
        """
        if volatility is not None:
            self.market_state['volatility'] = volatility
        
        if trend is not None:
            self.market_state['trend'] = trend
        
        if correlation is not None:
            self.market_state['correlation'] = correlation
        
        if liquidity is not None:
            self.market_state['liquidity'] = liquidity
        
        if sentiment is not None:
            self.market_state['sentiment'] = sentiment
        
        # Determine overall market risk level based on state
        self._update_risk_level()
        
        logger.info(f"Market state updated: Risk Level: {self.risk_level.name}")
    
    def update_position(self, symbol, current_price, size=None, cost_basis=None, timestamp=None):
        """
        Update or add a position in the portfolio
        
        Parameters:
          - symbol: Asset symbol
          - current_price: Current price of the asset
          - size: Position size (in shares or units)
          - cost_basis: Average entry price
          - timestamp: Optional timestamp (defaults to now)
        """
        timestamp = timestamp or datetime.datetime.now()
        
        # If position exists, update it
        if symbol in self.positions:
            position = self.positions[symbol]
            
            if size is not None:
                position['size'] = size
            
            if cost_basis is not None:
                position['cost_basis'] = cost_basis
            
            # Update current value and P&L
            position['current_price'] = current_price
            position['current_value'] = position['size'] * current_price
            position['unrealized_pnl'] = position['current_value'] - (position['size'] * position['cost_basis'])
            position['unrealized_pnl_pct'] = (position['unrealized_pnl'] / (position['size'] * position['cost_basis'])) if position['size'] > 0 else 0
            position['last_update'] = timestamp
            
        # Otherwise create a new position
        elif size is not None and cost_basis is not None:
            self.positions[symbol] = {
                'symbol': symbol,
                'size': size,
                'cost_basis': cost_basis,
                'current_price': current_price,
                'current_value': size * current_price,
                'unrealized_pnl': size * (current_price - cost_basis),
                'unrealized_pnl_pct': (current_price / cost_basis - 1) if cost_basis > 0 else 0,
                'entry_date': timestamp,
                'last_update': timestamp,
                'stop_loss': cost_basis * (1 - self.stop_loss_pct),
                'take_profit': cost_basis * (1 + self.take_profit_pct)
            }
        
        logger.debug(f"Position updated: {symbol}, Size: {size}, Value: ${size * current_price if size else 0:.2f}")
    
    def close_position(self, symbol, exit_price, timestamp=None, reason="Manual"):
        """
        Close a position and record the trade outcome
        
        Parameters:
          - symbol: Asset symbol
          - exit_price: Exit price
          - timestamp: Optional timestamp (defaults to now)
          - reason: Reason for closing the position
          
        Returns:
          - dict: Trade details
        """
        timestamp = timestamp or datetime.datetime.now()
        
        if symbol not in self.positions:
            logger.warning(f"Attempted to close non-existent position: {symbol}")
            return None
        
        position = self.positions[symbol]
        
        # Calculate realized P&L
        realized_pnl = position['size'] * (exit_price - position['cost_basis'])
        realized_pnl_pct = (exit_price / position['cost_basis'] - 1) if position['cost_basis'] > 0 else 0
        
        # Record the trade
        trade = {
            'symbol': symbol,
            'entry_date': position['entry_date'],
            'exit_date': timestamp,
            'entry_price': position['cost_basis'],
            'exit_price': exit_price,
            'size': position['size'],
            'realized_pnl': realized_pnl,
            'realized_pnl_pct': realized_pnl_pct,
            'duration': (timestamp - position['entry_date']).days,
            'reason': reason
        }
        
        self.trade_history.append(trade)
        self.total_pnl += realized_pnl
        
        # Update win/loss counts
        if realized_pnl > 0:
            self.win_count += 1
        else:
            self.loss_count += 1
        
        # Remove the position
        del self.positions[symbol]
        
        logger.info(f"Position closed: {symbol}, P&L: ${realized_pnl:.2f} ({realized_pnl_pct:.2%}), Reason: {reason}")
        
        return trade
    
    def check_stop_loss(self, symbol, current_price):
        """
        Check if the position has hit its stop loss level
        
        Parameters:
          - symbol: Asset symbol
          - current_price: Current price
          
        Returns:
          - bool: True if stop loss hit
        """
        if symbol not in self.positions:
            return False
        
        position = self.positions[symbol]
        
        # Check if price is below stop loss
        if current_price <= position['stop_loss']:
            logger.info(f"Stop loss triggered for {symbol} at {current_price:.2f}")
            return True
        
        return False
    
    def check_take_profit(self, symbol, current_price):
        """
        Check if the position has hit its take profit level
        
        Parameters:
          - symbol: Asset symbol
          - current_price: Current price
          
        Returns:
          - bool: True if take profit hit
        """
        if symbol not in self.positions:
            return False
        
        position = self.positions[symbol]
        
        # Check if price is above take profit
        if current_price >= position['take_profit']:
            logger.info(f"Take profit triggered for {symbol} at {current_price:.2f}")
            return True
        
        return False
    
    def adjust_stop_loss(self, symbol, new_stop_loss=None, trailing_pct=None):
        """
        Adjust the stop loss for a position
        
        Parameters:
          - symbol: Asset symbol
          - new_stop_loss: New stop loss price (absolute)
          - trailing_pct: Trailing stop % below highest price
          
        Returns:
          - float: The new stop loss level
        """
        if symbol not in self.positions:
            logger.warning(f"Cannot adjust stop loss for non-existent position: {symbol}")
            return None
        
        position = self.positions[symbol]
        current_price = position['current_price']
        
        # Apply trailing stop if specified
        if trailing_pct is not None:
            # Calculate new stop loss as percentage below current price
            new_sl = current_price * (1 - trailing_pct)
            
            # Only adjust if the new stop loss is higher than the current one
            if new_sl > position['stop_loss']:
                position['stop_loss'] = new_sl
                logger.info(f"Trailing stop adjusted for {symbol}: {new_sl:.2f}")
                
        # Apply absolute stop loss if specified
        elif new_stop_loss is not None:
            position['stop_loss'] = new_stop_loss
            logger.info(f"Stop loss adjusted for {symbol}: {new_stop_loss:.2f}")
        
        return position['stop_loss']
    
    def calculate_position_size(self, symbol, price, volatility=None, win_probability=None, expected_return=None):
        """
        Calculate optimal position size based on the selected sizing method
        
        Parameters:
          - symbol: Asset symbol
          - price: Current price
          - volatility: Asset's historical volatility (std dev of returns)
          - win_probability: Estimated probability of winning the trade
          - expected_return: Expected return of the trade
          
        Returns:
          - tuple: (dollar_amount, number_of_shares)
        """
        if self.portfolio_value <= 0:
            logger.warning("Cannot calculate position size with zero portfolio value")
            return 0, 0
            
        # Apply risk level modifiers
        risk_level_modifier = self._get_risk_level_modifier()
        
        # Equal weighting method
        if self.position_sizing_method == 'equal':
            # Simple equal weighting with risk level adjustment
            base_weight = min(1.0 / max(len(self.positions) + 1, 1), self.max_position_size)
            adjusted_weight = base_weight * risk_level_modifier
            
            # Ensure minimum position size
            dollar_amount = max(self.portfolio_value * adjusted_weight, 
                               self.portfolio_value * self.min_position_size)
            
        # Volatility-based position sizing
        elif self.position_sizing_method == 'volatility':
            if volatility is None or volatility <= 0:
                logger.warning(f"Invalid volatility for {symbol}: {volatility}")
                volatility = 0.01  # Default volatility assumption
            
            # Target risk = portfolio value * max risk per trade
            target_risk_dollars = self.portfolio_value * self.max_portfolio_risk * risk_level_modifier
            
            # Position size = target risk / volatility
            dollar_amount = target_risk_dollars / volatility
            
            # Apply position size limits
            dollar_amount = min(dollar_amount, self.portfolio_value * self.max_position_size)
            dollar_amount = max(dollar_amount, self.portfolio_value * self.min_position_size)
            
        # Kelly criterion position sizing
        elif self.position_sizing_method == 'kelly':
            if win_probability is None or expected_return is None:
                logger.warning(f"Missing win probability or expected return for Kelly calculation")
                # Fall back to volatility-based sizing
                return self.calculate_position_size(symbol, price, volatility, None, None)
            
            # Calculate Kelly fraction: f = (bp - q) / b
            # where b = win/loss ratio, p = win probability, q = loss probability
            win_loss_ratio = expected_return / self.stop_loss_pct
            loss_probability = 1 - win_probability
            
            kelly_fraction = (win_loss_ratio * win_probability - loss_probability) / win_loss_ratio
            
            # Apply fractional Kelly (more conservative)
            kelly_fraction = kelly_fraction * self.kelly_fraction
            
            # Ensure kelly is non-negative and within bounds
            kelly_fraction = max(0, min(kelly_fraction, self.max_position_size))
            
            # Apply risk level modifier
            kelly_fraction = kelly_fraction * risk_level_modifier
            
            # Calculate dollar amount
            dollar_amount = self.portfolio_value * kelly_fraction
            
            # Ensure minimum position size
            dollar_amount = max(dollar_amount, self.portfolio_value * self.min_position_size)
            
        else:
            logger.warning(f"Unknown position sizing method: {self.position_sizing_method}")
            dollar_amount = self.portfolio_value * self.min_position_size
        
        # Calculate number of shares
        if price <= 0:
            logger.warning(f"Invalid price for {symbol}: {price}")
            num_shares = 0
        else:
            num_shares = int(dollar_amount / price)
        
        logger.info(f"Position size for {symbol}: ${dollar_amount:.2f}, {num_shares} shares")
        
        return dollar_amount, num_shares
    
    def evaluate_signal(self, symbol, signal_type, signal_strength, price, 
                       volatility=None, win_probability=None, expected_return=None):
        """
        Evaluate a trading signal and determine whether to act on it
        
        Parameters:
          - symbol: Asset symbol
          - signal_type: Type of signal (ENTRY, EXIT, ADJUST)
          - signal_strength: Strength of the signal (-1 to 1)
          - price: Current price
          - volatility: Asset volatility
          - win_probability: Estimated win probability
          - expected_return: Expected return
          
        Returns:
          - dict: Decision with position size and reason
        """
        # Check circuit breakers first
        if self._check_circuit_breakers():
            return {
                'action': 'reject',
                'reason': f"Circuit breaker active: {self.risk_level.name} risk level"
            }
        
        # For entry signals
        if signal_type == SignalType.ENTRY:
            # Don't enter new positions during high risk
            if self.risk_level >= RiskLevel.HIGH:
                return {
                    'action': 'reject',
                    'reason': f"Risk level too high: {self.risk_level.name}"
                }
            
            # Check if we already have a position
            if symbol in self.positions:
                return {
                    'action': 'reject',
                    'reason': f"Position for {symbol} already exists"
                }
            
            # Check if signal is strong enough
            min_threshold = 0.3 + (0.2 * (self.risk_level.value - 1))  # Higher threshold in higher risk
            if abs(signal_strength) < min_threshold:
                return {
                    'action': 'reject',
                    'reason': f"Signal strength insufficient: {signal_strength:.2f} < {min_threshold:.2f}"
                }
            
            # Calculate position size
            dollar_amount, num_shares = self.calculate_position_size(
                symbol, price, volatility, win_probability, expected_return
            )
            
            if num_shares <= 0:
                return {
                    'action': 'reject',
                    'reason': "Calculated position size too small"
                }
            
            # Determine entry parameters
            stop_loss = price * (1 - self.stop_loss_pct)
            take_profit = price * (1 + self.take_profit_pct)
            
            return {
                'action': 'enter',
                'size': num_shares,
                'price': price,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'value': dollar_amount,
                'direction': 'long' if signal_strength > 0 else 'short'
            }
            
        # For exit signals
        elif signal_type == SignalType.EXIT:
            # Check if we have a position
            if symbol not in self.positions:
                return {
                    'action': 'reject',
                    'reason': f"No position for {symbol} exists"
                }
            
            position = self.positions[symbol]
            
            # Force exit on extreme risk
            if self.risk_level == RiskLevel.EXTREME:
                return {
                    'action': 'exit',
                    'reason': f"Forced exit due to extreme risk level",
                    'price': price,
                    'size': position['size']
                }
            
            # Check if signal is strong enough for exit
            min_threshold = 0.2  # Lower threshold for exits
            if abs(signal_strength) < min_threshold:
                # Still check stops
                if self.check_stop_loss(symbol, price):
                    return {
                        'action': 'exit',
                        'reason': 'Stop loss hit',
                        'price': price,
                        'size': position['size']
                    }
                
                if self.check_take_profit(symbol, price):
                    return {
                        'action': 'exit',
                        'reason': 'Take profit hit',
                        'price': price,
                        'size': position['size']
                    }
                
                return {
                    'action': 'reject',
                    'reason': f"Exit signal strength insufficient: {signal_strength:.2f} < {min_threshold:.2f}"
                }
            
            return {
                'action': 'exit',
                'reason': f"Exit signal triggered",
                'price': price,
                'size': position['size']
            }
            
        # For position adjustment signals
        elif signal_type == SignalType.ADJUST:
            # Check if we have a position
            if symbol not in self.positions:
                return {
                    'action': 'reject',
                    'reason': f"No position for {symbol} exists"
                }
            
            position = self.positions[symbol]
            
            # Calculate new stop loss (trailing if in profit)
            if position['unrealized_pnl'] > 0:
                # Calculate trailing stop based on signal strength
                trailing_pct = min(self.stop_loss_pct, self.stop_loss_pct * (1 - signal_strength * 0.5))
                new_stop = price * (1 - trailing_pct)
                
                # Only adjust if it raises the stop
                if new_stop > position['stop_loss']:
                    return {
                        'action': 'adjust',
                        'stop_loss': new_stop,
                        'reason': 'Trailing stop adjustment'
                    }
            
            return {
                'action': 'reject',
                'reason': 'No adjustment needed'
            }
        
        else:
            logger.warning(f"Unknown signal type: {signal_type}")
            return {
                'action': 'reject',
                'reason': f"Unknown signal type: {signal_type}"
            }
    
    def get_portfolio_stats(self):
        """
        Get comprehensive portfolio statistics
        
        Returns:
          - dict: Portfolio statistics
        """
        # Calculate portfolio value
        portfolio_value = self.portfolio_value
        positions_value = sum(p['current_value'] for p in self.positions.values())
        cash = portfolio_value - positions_value
        
        # Calculate exposure
        exposure = positions_value / portfolio_value if portfolio_value > 0 else 0
        
        # Calculate drawdown
        peak_value = self.max_portfolio_value
        drawdown = 1 - (portfolio_value / peak_value) if peak_value > 0 else 0
        
        # Calculate win rate
        total_trades = self.win_count + self.loss_count
        win_rate = self.win_count / total_trades if total_trades > 0 else 0
        
        # Calculate position diversification
        num_positions = len(self.positions)
        largest_position = max([p['current_value'] for p in self.positions.values()], default=0)
        largest_position_pct = largest_position / portfolio_value if portfolio_value > 0 else 0
        
        # Calculate volatility of portfolio
        if len(self.portfolio_history) > 2:
            values = [entry['value'] for entry in self.portfolio_history[-30:]]
            returns = [values[i] / values[i-1] - 1 for i in range(1, len(values))]
            portfolio_volatility = np.std(returns) if returns else 0
        else:
            portfolio_volatility = 0
        
        return {
            'portfolio_value': portfolio_value,
            'positions_value': positions_value,
            'cash': cash,
            'exposure': exposure,
            'drawdown': drawdown,
            'num_positions': num_positions,
            'largest_position_pct': largest_position_pct,
            'win_rate': win_rate,
            'total_trades': total_trades,
            'total_pnl': self.total_pnl,
            'portfolio_volatility': portfolio_volatility,
            'risk_level': self.risk_level.name
        }
    
    def _check_circuit_breakers(self):
        """
        Check if any circuit breakers should be triggered
        
        Returns:
          - bool: True if circuit breaker active
        """
        # Check drawdown circuit breaker
        if self.current_drawdown >= self.max_drawdown_limit:
            logger.warning(f"Circuit breaker triggered: Max drawdown limit exceeded ({self.current_drawdown:.2%})")
            self.risk_level = RiskLevel.EXTREME
            return True
        
        # Check volatility circuit breaker
        if self.market_state['volatility'] >= 0.5:
            logger.warning(f"Circuit breaker triggered: Excessive volatility ({self.market_state['volatility']:.2f})")
            self._update_risk_level()
            return self.risk_level >= RiskLevel.HIGH
        
        return False
    
    def _update_risk_level(self):
        """Update overall risk level based on market state"""
        # Calculate a composite risk score
        risk_score = (
            self.market_state['volatility'] * 0.5 +
            abs(self.market_state['trend']) * 0.1 +
            self.market_state['correlation'] * 0.2 +
            (1 - self.market_state['liquidity']) * 0.1 +
            abs(self.market_state['sentiment']) * 0.1
        )
        
        # Determine risk level based on score
        if risk_score < 0.2:
            self.risk_level = RiskLevel.LOW
        elif risk_score < 0.4:
            self.risk_level = RiskLevel.MEDIUM
        elif risk_score < 0.6:
            self.risk_level = RiskLevel.HIGH
        else:
            self.risk_level = RiskLevel.EXTREME
        
        logger.info(f"Risk level updated to {self.risk_level.name} (score: {risk_score:.2f})")
    
    def _get_risk_level_modifier(self):
        """Get position size modifier based on risk level"""
        if self.risk_level == RiskLevel.LOW:
            return 1.0
        elif self.risk_level == RiskLevel.MEDIUM:
            return 0.75
        elif self.risk_level == RiskLevel.HIGH:
            return 0.5
        else:  # EXTREME
            return 0.25

if __name__ == '__main__':
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Example usage
    risk_manager = RiskManager(
        position_sizing_method='volatility',
        max_position_size=0.2,
        max_portfolio_risk=0.01,
        max_drawdown_limit=0.15
    )
    
    # Initialize portfolio
    risk_manager.update_portfolio_value(100000)
    
    # Simulate market state update
    risk_manager.update_market_state(
        volatility=0.15,
        trend=0.2,
        correlation=0.3,
        liquidity=0.9,
        sentiment=0.1
    )
    
    # Calculate position size
    amount, shares = risk_manager.calculate_position_size('AAPL', 150, volatility=0.02)
    print(f"Position size for AAPL: ${amount:.2f}, {shares} shares")
    
    # Evaluate a trading signal
    result = risk_manager.evaluate_signal(
        symbol='AAPL', 
        signal_type=SignalType.ENTRY, 
        signal_strength=0.7, 
        price=150,
        volatility=0.02,
        win_probability=0.6,
        expected_return=0.05
    )
    print(f"Signal evaluation result: {result}")
    
    # Update a position
    if result['action'] == 'enter':
        risk_manager.update_position(
            symbol='AAPL',
            current_price=150,
            size=result['size'],
            cost_basis=150
        )
    
    # Get portfolio stats
    stats = risk_manager.get_portfolio_stats()
    print(f"Portfolio stats: {stats}")