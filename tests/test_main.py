import unittest
import os
import json
import pandas as pd
import numpy as np
from modules.data_acquisition import fetch_price_data
from modules.ml_predictor import ensemble_predict_wrapper
from modules.sentiment_analysis import analyze_sentiment
from modules.signal_aggregator import aggregate_signals
from modules.risk_manager import RiskManager, SignalType
from modules.broker_integration import BrokerIntegration

class TestMainWorkflow(unittest.TestCase):
    
    def setUp(self):
        """Set up for integration testing"""
        self.symbols = ['AAPL', 'MSFT']
        self.start_date = '2023-01-01'
        self.end_date = '2023-01-31'
        self.risk_manager = RiskManager()
        self.broker = BrokerIntegration(paper=True)
    
    def test_full_workflow(self):
        """Test the full workflow from data acquisition to signal generation"""
        try:
            # 1. Fetch data
            price_data = {}
            for symbol in self.symbols:
                data = fetch_price_data(symbol, self.start_date, self.end_date)
                self.assertIsNotNone(data)
                self.assertFalse(data.empty)
                price_data[symbol] = data
            
            # 2. Prepare features and generate predictions
            for symbol, data in price_data.items():
                # Prepare train/test split
                train_data = data.iloc[:-5].copy()
                test_data = data.iloc[-5:].copy()
                
                # Add target
                train_data['target'] = train_data['Close'].pct_change().shift(-1)
                train_data = train_data.dropna()
                
                X_train = train_data.drop(columns=['target'])
                y_train = train_data['target']
                X_test = test_data
                
                # Generate predictions
                predictions = ensemble_predict_wrapper(X_train, y_train, X_test)
                self.assertIsNotNone(predictions)
                self.assertEqual(len(predictions), len(X_test))
                
                # 3. Generate sentiment
                sentiment_result = analyze_sentiment(f"Latest news for {symbol}")
                self.assertIsNotNone(sentiment_result)
                self.assertIn('score', sentiment_result)
                
                # 4. Aggregate signals
                signal = aggregate_signals(
                    f"Latest news for {symbol}",
                    sentiment_result['score'],
                    symbol,
                    X_train,
                    y_train,
                    X_test,
                    [1, 1, 5, 10, 2]
                )
                self.assertIsNotNone(signal)
                
                # 5. Evaluate with risk manager
                if signal > 0.5:  # Positive signal
                    result = self.risk_manager.evaluate_signal(
                        symbol=symbol,
                        signal_type=SignalType.ENTRY,
                        signal_strength=signal,
                        price=data['Close'].iloc[-1]
                    )
                    self.assertIsNotNone(result)
                    self.assertIn('action', result)
                
                # 6. Execute trade if appropriate
                if signal > 0.5:
                    signals = {
                        symbol: {
                            'action': 'enter',
                            'signal': signal,
                            'price': data['Close'].iloc[-1]
                        }
                    }
                    execution_result = self.broker.execute_trades(signals, self.risk_manager)
                    self.assertIsNotNone(execution_result)
        
        except Exception as e:
            self.fail(f"Full workflow test failed: {str(e)}")
    
    def tearDown(self):
        """Clean up after tests"""
        # Reset broker state
        self.broker = None