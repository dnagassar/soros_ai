# modules/risk_manager.py (improved version)
import numpy as np
import pandas as pd
import logging
from enum import Enum
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class SignalType(Enum):
    """Enum for signal types"""
    ENTRY = 1
    EXIT = 2
    ADJUST = 3

class RiskManager:
    """
    Enhanced risk management system with improved portfolio-level risk controls
    and simplified parameter set
    """
    
    def __init__(self, 
                max_position_size=0.1,        # Maximum single position size (% of portfolio)
                max_portfolio_risk=0.02,      # Maximum portfolio-level daily VaR (% of portfolio)
                stop_loss_pct=0.03,           # Default stop loss percentage
                take_profit_pct=0.04,         # Default take profit percentage 
                max_sector_allocation=0.25,   # Maximum allocation to any sector
                max_correlation_exposure=0.4, # Maximum exposure to correlated assets
                position_sizing_method='volatility'): # 'equal', 'volatility', 'kelly'
        
        self.max_position_size = max_position_size
        self.max_portfolio_risk = max_portfolio_risk
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.max_sector_allocation = max_sector_allocation
        self.max_correlation_exposure = max_correlation_exposure
        self.position_sizing_method = position_sizing_method
        
        # Portfolio state tracking
        self.portfolio_value = 0.0
        self.positions = {}           # Symbol -> position details
        self.portfolio_history = []   # Historical portfolio values
        self.sector_allocations = {}  # Sector -> allocation percentage
        self.correlation_matrix = {}  # Symbol pair -> correlation value
        
        # Performance tracking
        self.trade_history = []
        self.win_count = 0
        self.loss_count = 0
        self.total_pnl = 0.0
        
        logger.info(f"Risk Manager initialized with position sizing method: {position_sizing_method}")
    
    def update_portfolio_value(self, new_value, timestamp=None):
        """Update the current portfolio value and track history"""
        timestamp = timestamp or datetime.now()
        
        self.portfolio_value = new_value
        self.portfolio_history.append({
            'timestamp': timestamp,
            'value': new_value
        })
        
        # Calculate drawdown
        peak_value = max([entry['value'] for entry in self.portfolio_history]) if self.portfolio_history else new_value
        current_drawdown = 1 - (new_value / peak_value) if peak_value > 0 else 0
        
        # Update internal state
        self._update_sector_allocations()
        self._update_correlation_exposures()
        
        logger.debug(f"Portfolio value updated: ${new_value:.2f}, Drawdown: {current_drawdown:.2%}")
    
    def _update_sector_allocations(self):
        """Update sector allocations based on current positions"""
        self.sector_allocations = {}
        
        # Skip if no positions or portfolio value is zero
        if not self.positions or self.portfolio_value <= 0:
            return
            
        # Group positions by sector
        for symbol, position in self.positions.items():
            sector = position.get('sector', 'Unknown')
            position_value = position.get('current_value', 0)
            
            if sector not in self.sector_allocations:
                self.sector_allocations[sector] = 0
                
            self.sector_allocations[sector] += position_value / self.portfolio_value
    
    def _update_correlation_exposures(self):
        """Update correlation exposures between positions"""
        # This would use historical return data to calculate correlations
        # For now, we'll use a simplified approach
        symbols = list(self.positions.keys())
        
        for i, symbol1 in enumerate(symbols):
            for symbol2 in symbols[i+1:]:
                # In a real implementation, calculate actual correlation
                # For now, assume moderate correlation between different assets
                self.correlation_matrix[(symbol1, symbol2)] = 0.3
    
    def calculate_position_size(self, symbol, price, volatility, sector=None):
        """
        Calculate position size with enhanced risk controls
        
        Returns:
          - tuple: (dollar_amount, number_of_shares)
        """
        if self.portfolio_value <= 0 or price <= 0:
            return 0, 0
        
        # Base position size using the selected method
        if self.position_sizing_method == 'equal':
            # Equal position sizing
            dollar_amount = self.portfolio_value * self.max_position_size
            
        elif self.position_sizing_method == 'volatility':
            # Volatility-based position sizing
            if volatility <= 0:
                volatility = 0.01  # Default if volatility is invalid
                
            # Target risk per position
            risk_dollars = self.portfolio_value * self.max_portfolio_risk * 0.1  # 10% of total risk budget
            
            # Position size = risk / volatility
            dollar_amount = risk_dollars / volatility
            
            # Apply position size limits
            dollar_amount = min(dollar_amount, self.portfolio_value * self.max_position_size)
            
        else:  # Default to equal weighting if method not recognized
            dollar_amount = self.portfolio_value * self.max_position_size
        
        # Apply sector constraints if sector is provided
        if sector:
            current_sector_allocation = self.sector_allocations.get(sector, 0)
            if current_sector_allocation >= self.max_sector_allocation:
                # Sector allocation would be exceeded, reduce position size
                logger.info(f"Reducing position size due to sector constraint for {sector}")
                return 0, 0
            
            # Limit position to keep sector under max allocation
            max_additional_allocation = self.max_sector_allocation - current_sector_allocation
            max_sector_dollars = self.portfolio_value * max_additional_allocation
            dollar_amount = min(dollar_amount, max_sector_dollars)
        
        # Apply correlation constraints
        if self._check_correlation_exposure(symbol) > self.max_correlation_exposure:
            logger.info(f"Reducing position size due to correlation constraint for {symbol}")
            dollar_amount *= 0.5  # Reduce position size by half
        
        # Calculate number of shares
        num_shares = int(dollar_amount / price)
        
        return dollar_amount, num_shares
    
    def _check_correlation_exposure(self, new_symbol):
        """
        Check correlation exposure if adding a new position
        
        Returns:
          - float: Correlation exposure (0-1)
        """
        # Simplified version - in a real implementation, calculate actual exposure
        # based on correlation matrix and position sizes
        return 0.2  # Default moderate exposure
    
    def evaluate_signal(self, symbol, signal_type, signal_strength, price, 
                       volatility=None, sector=None):
        """
        Evaluate a trading signal with enhanced risk checks
        
        Returns:
          - dict: Decision with position size and reason
        """
        # For entry signals
        if signal_type == SignalType.ENTRY:
            # Check if we already have a position
            if symbol in self.positions:
                return {
                    'action': 'reject',
                    'reason': f"Position for {symbol} already exists"
                }
            
            # Check if signal is strong enough
            min_threshold = 0.3
            if abs(signal_strength) < min_threshold:
                return {
                    'action': 'reject',
                    'reason': f"Signal strength insufficient: {signal_strength:.2f} < {min_threshold:.2f}"
                }
            
            # Check portfolio capacity - are we fully invested?
            current_exposure = sum(p.get('current_value', 0) for p in self.positions.values()) / max(1, self.portfolio_value)
            if current_exposure > 0.9:  # Allow 90% max exposure
                return {
                    'action': 'reject',
                    'reason': "Portfolio capacity reached"
                }
            
            # Calculate position size with all constraints
            dollar_amount, num_shares = self.calculate_position_size(
                symbol, price, volatility or 0.02, sector
            )
            
            if num_shares <= 0:
                return {
                    'action': 'reject',
                    'reason': "Position sizing resulted in zero shares"
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
            
            # Check if signal is strong enough for exit or stop loss hit
            exit_threshold = -0.2  # Lower threshold for exits
            if signal_strength < exit_threshold or price <= position.get('stop_loss', 0):
                return {
                    'action': 'exit',
                    'reason': 'Exit signal triggered or stop loss hit',
                    'price': price,
                    'size': position.get('size', 0)
                }
                
            # Check take profit
            if price >= position.get('take_profit', float('inf')):
                return {
                    'action': 'exit',
                    'reason': 'Take profit reached',
                    'price': price,
                    'size': position.get('size', 0)
                }
            
            return {
                'action': 'reject',
                'reason': f"Exit criteria not met"
            }
            
        else:
            logger.warning(f"Unknown signal type: {signal_type}")
            return {
                'action': 'reject',
                'reason': f"Unknown signal type: {signal_type}"
            }
    
    def update_position(self, symbol, current_price, size=None, cost_basis=None, 
                      sector=None, timestamp=None):
        """Update or add a position with improved tracking"""
        timestamp = timestamp or datetime.now()
        
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
                'take_profit': cost_basis * (1 + self.take_profit_pct),
                'sector': sector or 'Unknown'
            }
            
        # Update portfolio state after position change
        self._update_sector_allocations()
    
    def close_position(self, symbol, exit_price, timestamp=None, reason="Manual"):
        """Close a position and record the trade outcome"""
        timestamp = timestamp or datetime.now()
        
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
            'duration_days': (timestamp - position['entry_date']).days,
            'sector': position.get('sector', 'Unknown'),
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
        
        # Update portfolio state
        self._update_sector_allocations()
        
        logger.info(f"Position closed: {symbol}, P&L: ${realized_pnl:.2f} ({realized_pnl_pct:.2%}), Reason: {reason}")
        
        return trade
    
    def get_portfolio_stats(self):
        """Get comprehensive portfolio statistics"""
        # Calculate portfolio value
        positions_value = sum(p.get('current_value', 0) for p in self.positions.values())
        cash = self.portfolio_value - positions_value
        
        # Calculate exposure
        exposure = positions_value / self.portfolio_value if self.portfolio_value > 0 else 0
        
        # Calculate drawdown
        peak_value = max([entry['value'] for entry in self.portfolio_history]) if self.portfolio_history else self.portfolio_value
        drawdown = 1 - (self.portfolio_value / peak_value) if peak_value > 0 else 0
        
        # Calculate win rate
        total_trades = self.win_count + self.loss_count
        win_rate = self.win_count / total_trades if total_trades > 0 else 0
        
        # Calculate sector exposures
        sector_exposure = self.sector_allocations.copy()
        
        # Calculate concentration metrics
        num_positions = len(self.positions)
        largest_position = max([p.get('current_value', 0) for p in self.positions.values()], default=0)
        largest_position_pct = largest_position / self.portfolio_value if self.portfolio_value > 0 else 0
        
        # Calculate portfolio beta (simplified)
        portfolio_beta = 1.0  # Default to market beta
        
        return {
            'portfolio_value': self.portfolio_value,
            'positions_value': positions_value,
            'cash': cash,
            'exposure': exposure,
            'drawdown': drawdown,
            'num_positions': num_positions,
            'largest_position_pct': largest_position_pct,
            'win_rate': win_rate,
            'total_trades': total_trades,
            'total_pnl': self.total_pnl,
            'sector_exposure': sector_exposure,
            'portfolio_beta': portfolio_beta
        }