# Create modules/broker_integration.py
import logging
import os
import pandas as pd
from datetime import datetime
from enum import Enum
import time
import traceback

logger = logging.getLogger(__name__)

class OrderType(Enum):
    MARKET = 1
    LIMIT = 2
    STOP = 3
    STOP_LIMIT = 4

class BrokerIntegration:
    """
    Broker integration module that connects to trading APIs
    for executing orders and managing positions.
    """
    
    def __init__(self, api_key=None, api_secret=None, paper=True):
        """Initialize broker connection"""
        self.api_key = api_key
        self.api_secret = api_secret
        self.paper = paper
        self.client = None
        
        # Paper trading simulation data
        self.paper_account = {
            'cash': 100000.0,
            'equity': 100000.0,
            'buying_power': 100000.0,
            'positions': {},
            'orders': [],
            'trades': []
        }
        
        self._connect()
        
    def _connect(self):
        """Establish connection to broker API"""
        try:
            # Try to import and connect to Alpaca if keys are available
            if self.api_key and self.api_secret:
                try:
                    from alpaca.trading.client import TradingClient
                    self.client = TradingClient(self.api_key, self.api_secret, paper=self.paper)
                    logger.info("Connected to Alpaca API")
                except ImportError:
                    logger.warning("Alpaca SDK not installed. Using paper trading simulation.")
                    self.client = None
                except Exception as e:
                    logger.error(f"Failed to connect to Alpaca API: {e}")
                    self.client = None
            else:
                logger.info("No API credentials provided. Using paper trading simulation.")
        except Exception as e:
            logger.error(f"Error in broker initialization: {e}")
            self.client = None
    
    def check_connection(self):
        """Check if broker connection is active"""
        if not self.client:
            # In paper simulation mode, we're always "connected"
            return self.paper
            
        try:
            # For Alpaca, try to get account info
            account = self.client.get_account()
            return True
        except Exception:
            return False
    
    def get_account_info(self):
        """Get account information"""
        if not self.client:
            # Return simulated account data
            total_position_value = sum(
                pos['qty'] * pos['current_price'] 
                for pos in self.paper_account['positions'].values()
            )
            
            self.paper_account['equity'] = self.paper_account['cash'] + total_position_value
            self.paper_account['buying_power'] = self.paper_account['cash'] * 2  # 2x margin
            
            return {
                'id': 'paper-account',
                'cash': self.paper_account['cash'],
                'equity': self.paper_account['equity'],
                'buying_power': self.paper_account['buying_power'],
                'long_market_value': total_position_value,
                'short_market_value': 0.0,
                'initial_margin': 0.0,
                'maintenance_margin': 0.0,
                'last_equity': self.paper_account['equity'],
                'status': 'ACTIVE'
            }
            
        try:
            # For Alpaca, get account info from API
            account = self.client.get_account()
            return {
                'id': account.id,
                'cash': float(account.cash),
                'equity': float(account.equity),
                'buying_power': float(account.buying_power),
                'long_market_value': float(account.long_market_value),
                'short_market_value': float(account.short_market_value),
                'initial_margin': float(account.initial_margin),
                'maintenance_margin': float(account.maintenance_margin),
                'last_equity': float(account.last_equity),
                'status': account.status
            }
        except Exception as e:
            logger.error(f"Error getting account info: {e}")
            return None
    
    def get_positions(self):
        """Get current positions"""
        if not self.client:
            # Return simulated positions
            return [
                {
                    'symbol': symbol,
                    'qty': float(position['qty']),
                    'market_value': float(position['qty'] * position['current_price']),
                    'cost_basis': float(position['cost_basis']),
                    'unrealized_pl': float(position['qty'] * (position['current_price'] - position['cost_basis'])),
                    'unrealized_plpc': float((position['current_price'] / position['cost_basis'] - 1) * 100) if position['cost_basis'] > 0 else 0.0,
                    'current_price': float(position['current_price']),
                    'last_day_price': float(position.get('last_day_price', position['current_price'])),
                    'change_today': float((position['current_price'] / position.get('last_day_price', position['current_price']) - 1) * 100) if position.get('last_day_price', 0) > 0 else 0.0
                }
                for symbol, position in self.paper_account['positions'].items()
            ]
            
        try:
            # For Alpaca, get positions from API
            positions = self.client.get_all_positions()
            return [{
                'symbol': position.symbol,
                'qty': float(position.qty),
                'market_value': float(position.market_value),
                'cost_basis': float(position.cost_basis),
                'unrealized_pl': float(position.unrealized_pl),
                'unrealized_plpc': float(position.unrealized_plpc),
                'current_price': float(position.current_price),
                'last_day_price': float(position.lastday_price),
                'change_today': float(position.change_today)
            } for position in positions]
        except Exception as e:
            logger.error(f"Error getting positions: {e}")
            return []
    
    def place_order(self, symbol, quantity, side, order_type=OrderType.MARKET, limit_price=None, 
                   stop_price=None, time_in_force='day'):
        """
        Place an order
        
        Parameters:
          - symbol: Asset symbol
          - quantity: Order quantity
          - side: 'buy' or 'sell'
          - order_type: OrderType enum
          - limit_price: Limit price for limit orders
          - stop_price: Stop price for stop orders
          - time_in_force: Order time in force
          
        Returns:
          - dict: Order information
        """
        if not self.client:
            # Simulate order placement
            try:
                # Generate order ID
                order_id = f"paper-{int(time.time())}-{len(self.paper_account['orders'])}"
                
                # Get current market price (simulated)
                # In a real implementation, you would fetch this from market data
                current_price = 0
                
                # If we have a position, use its current price
                if symbol in self.paper_account['positions']:
                    current_price = self.paper_account['positions'][symbol]['current_price']
                # If we have limit price, use that
                elif limit_price:
                    current_price = limit_price
                # Otherwise, simulate a price
                else:
                    # This is just a placeholder; you would replace this with actual price data
                    current_price = 100.0
                
                # Calculate order value
                order_value = quantity * current_price
                
                # Check if we have enough cash for a buy order
                if side.lower() == 'buy' and order_value > self.paper_account['cash']:
                    logger.warning(f"Insufficient funds for order: {order_value} > {self.paper_account['cash']}")
                    return None
                
                # Create order
                order = {
                    'id': order_id,
                    'client_order_id': f"client-{order_id}",
                    'symbol': symbol,
                    'quantity': float(quantity),
                    'side': side.lower(),
                    'type': order_type.name.lower(),
                    'status': 'filled',  # For simplicity, assume immediate fill
                    'created_at': datetime.now().isoformat(),
                    'filled_at': datetime.now().isoformat(),
                    'filled_price': current_price,
                    'limit_price': limit_price,
                    'stop_price': stop_price
                }
                
                self.paper_account['orders'].append(order)
                
                # Update positions and cash
                if side.lower() == 'buy':
                    # Reduce cash
                    self.paper_account['cash'] -= order_value
                    
                    # Update position
                    if symbol in self.paper_account['positions']:
                        # Update existing position
                        existing_qty = self.paper_account['positions'][symbol]['qty']
                        existing_cost = self.paper_account['positions'][symbol]['cost_basis'] * existing_qty
                        
                        new_qty = existing_qty + quantity
                        new_cost_basis = (existing_cost + order_value) / new_qty
                        
                        self.paper_account['positions'][symbol]['qty'] = new_qty
                        self.paper_account['positions'][symbol]['cost_basis'] = new_cost_basis
                        self.paper_account['positions'][symbol]['current_price'] = current_price
                    else:
                        # Create new position
                        self.paper_account['positions'][symbol] = {
                            'qty': quantity,
                            'cost_basis': current_price,
                            'current_price': current_price,
                            'last_day_price': current_price
                        }
                    
                    # Add trade to history
                    self.paper_account['trades'].append({
                        'symbol': symbol,
                        'side': 'buy',
                        'qty': quantity,
                        'price': current_price,
                        'value': order_value,
                        'timestamp': datetime.now().isoformat()
                    })
                
                elif side.lower() == 'sell':
                    # Check if we have the position
                    if symbol not in self.paper_account['positions'] or self.paper_account['positions'][symbol]['qty'] < quantity:
                        logger.warning(f"Insufficient shares for sell order: {symbol}")
                        return None
                    
                    # Update position
                    current_qty = self.paper_account['positions'][symbol]['qty']
                    cost_basis = self.paper_account['positions'][symbol]['cost_basis']
                    
                    # Calculate PnL
                    position_pnl = quantity * (current_price - cost_basis)
                    
                    # Update cash
                    self.paper_account['cash'] += order_value
                    
                    # Update position
                    remaining_qty = current_qty - quantity
                    
                    if remaining_qty > 0:
                        # Reduce position
                        self.paper_account['positions'][symbol]['qty'] = remaining_qty
                    else:
                        # Close position
                        del self.paper_account['positions'][symbol]
                    
                    # Add trade to history
                    self.paper_account['trades'].append({
                        'symbol': symbol,
                        'side': 'sell',
                        'qty': quantity,
                        'price': current_price,
                        'value': order_value,
                        'pnl': position_pnl,
                        'pnl_percent': (current_price / cost_basis - 1) * 100 if cost_basis > 0 else 0,
                        'timestamp': datetime.now().isoformat()
                    })
                
                return {
                    'id': order['id'],
                    'client_order_id': order['client_order_id'],
                    'symbol': order['symbol'],
                    'quantity': order['quantity'],
                    'side': order['side'],
                    'type': order['type'],
                    'status': order['status'],
                    'created_at': order['created_at']
                }
                
            except Exception as e:
                logger.error(f"Error in paper order simulation: {e}")
                logger.error(traceback.format_exc())
                return None
            
        try:
            # For Alpaca, use API for order placement
            from alpaca.trading.requests import MarketOrderRequest, LimitOrderRequest
            from alpaca.trading.enums import OrderSide, TimeInForce, OrderStatus
            
            # Convert side to Alpaca enum
            order_side = OrderSide.BUY if side.lower() == 'buy' else OrderSide.SELL
            
            # Convert time in force
            if time_in_force.lower() == 'day':
                time_in_force_enum = TimeInForce.DAY
            elif time_in_force.lower() == 'gtc':
                time_in_force_enum = TimeInForce.GTC
            else:
                time_in_force_enum = TimeInForce.DAY
            
            # Create order request based on type
            if order_type == OrderType.MARKET:
                order_request = MarketOrderRequest(
                    symbol=symbol,
                    qty=quantity,
                    side=order_side,
                    time_in_force=time_in_force_enum
                )
            elif order_type == OrderType.LIMIT and limit_price:
                order_request = LimitOrderRequest(
                    symbol=symbol,
                    qty=quantity,
                    side=order_side,
                    time_in_force=time_in_force_enum,
                    limit_price=limit_price
                )
            else:
                logger.error(f"Unsupported order type: {order_type}")
                return None
            
            # Submit order
            order = self.client.submit_order(order_request)
            
            # Return order details
            return {
                'id': order.id,
                'client_order_id': order.client_order_id,
                'symbol': order.symbol,
                'quantity': float(order.qty),
                'side': order.side.value,
                'type': order.type.value,
                'status': order.status.value,
                'created_at': order.created_at.isoformat() if order.created_at else None
            }
            
        except Exception as e:
            logger.error(f"Error placing order for {symbol}: {e}")
            return None
    
    def execute_trades(self, signals, risk_manager=None):
        """
        Execute trades based on signals
        
        Parameters:
          - signals: Dictionary of trading signals by symbol
          - risk_manager: Optional risk manager instance
          
        Returns:
          - dict: Execution results
        """
        execution_results = {
            'timestamp': datetime.now().isoformat(),
            'orders': [],
            'errors': []
        }
        
        # Get current positions
        current_positions = self.get_positions()
        position_symbols = {p['symbol']: p for p in current_positions}
        
        # Process signals
        for symbol, signal_data in signals.items():
            try:
                action = signal_data.get('action', 'unknown')
                strength = signal_data.get('signal', 0)
                
                # Skip if no clear action
                if action == 'unknown':
                    continue
                
                # Determine if we have an existing position
                has_position = symbol in position_symbols
                
                # Process entry signals
                if action == 'enter' and not has_position:
                    # Calculate position size
                    position_size = 0
                    
                    if risk_manager:
                        # Use risk manager to calculate position size
                        account_info = self.get_account_info()
                        if account_info:
                            from modules.risk_manager import SignalType
                            result = risk_manager.evaluate_signal(
                                symbol=symbol,
                                signal_type=SignalType.ENTRY,
                                signal_strength=strength,
                                price=signal_data.get('price', 0)
                            )
                            if result.get('action') == 'enter' and 'size' in result:
                                position_size = result['size']
                    
                    # If no size from risk manager, use simple sizing
                    if position_size <= 0:
                        account_info = self.get_account_info()
                        if account_info:
                            # Use 2% of account for each position
                            cash = float(account_info['cash'])
                            position_value = cash * 0.02
                            price = signal_data.get('price', 0)
                            if price > 0:
                                position_size = int(position_value / price)
                    
                    # Place order if size is valid
                    if position_size > 0:
                        order_result = self.place_order(
                            symbol=symbol,
                            quantity=position_size,
                            side='buy'
                        )
                        
                        if order_result:
                            execution_results['orders'].append(order_result)
                            logger.info(f"Placed buy order for {position_size} shares of {symbol}")
                
                # Process exit signals
                elif action == 'exit' and has_position:
                    # Get position details
                    position = position_symbols[symbol]
                    position_size = float(position['qty'])
                    
                    # Place sell order
                    if position_size > 0:
                        order_result = self.place_order(
                            symbol=symbol,
                            quantity=position_size,
                            side='sell'
                        )
                        
                        if order_result:
                            execution_results['orders'].append(order_result)
                            logger.info(f"Placed sell order for {position_size} shares of {symbol}")
            
            except Exception as e:
                error_msg = f"Error executing trade for {symbol}: {e}"
                logger.error(error_msg)
                execution_results['errors'].append(error_msg)
        
        execution_results['success'] = len(execution_results['errors']) == 0
        return execution_results