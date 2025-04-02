import unittest
from modules.broker_integration import BrokerIntegration, OrderType

class TestBrokerIntegration(unittest.TestCase):
    
    def setUp(self):
        """Set up broker in paper trading mode"""
        self.broker = BrokerIntegration(paper=True)
    
    def test_broker_initialization(self):
        """Test if broker can be initialized in paper mode"""
        self.assertIsNotNone(self.broker)
        self.assertTrue(self.broker.paper)
    
    def test_account_info(self):
        """Test if account info can be retrieved"""
        try:
            account_info = self.broker.get_account_info()
            
            self.assertIsNotNone(account_info)
            self.assertIn('cash', account_info)
            self.assertIn('equity', account_info)
            self.assertIn('buying_power', account_info)
        except Exception as e:
            self.fail(f"Failed to get account info: {str(e)}")
    
    def test_place_order(self):
        """Test if orders can be placed in paper mode"""
        try:
            order = self.broker.place_order(
                symbol='AAPL',
                quantity=10,
                side='buy',
                order_type=OrderType.MARKET
            )
            
            self.assertIsNotNone(order)
            self.assertIn('id', order)
            self.assertEqual(order['symbol'], 'AAPL')
            self.assertEqual(order['quantity'], 10.0)
            self.assertEqual(order['side'], 'buy')
        except Exception as e:
            self.fail(f"Failed to place order: {str(e)}")
    
    def test_execute_trades(self):
        """Test if trades can be executed based on signals"""
        # Create test signals
        signals = {
            'AAPL': {
                'action': 'enter',
                'signal': 0.8,
                'price': 150.0
            },
            'MSFT': {
                'action': 'exit',
                'signal': -0.6,
                'price': 280.0
            }
        }
        
        try:
            result = self.broker.execute_trades(signals)
            
            self.assertIsNotNone(result)
            self.assertIn('timestamp', result)
            self.assertIn('orders', result)
        except Exception as e:
            self.fail(f"Failed to execute trades: {str(e)}")