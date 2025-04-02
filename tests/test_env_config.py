import unittest
import importlib
import os
import sys
import json

class TestEnvironment(unittest.TestCase):
    
    def test_config_existence(self):
        """Test if config.py exists and can be imported"""
        try:
            import config
            self.assertTrue(True)
        except ImportError:
            self.fail("config.py file is missing or cannot be imported")
    
    def test_config_required_fields(self):
        """Test if config.py has all required fields"""
        try:
            import config
            required_fields = [
                'ALPHA_VANTAGE_API_KEY', 
                'OPENAI_API_KEY', 
                'FRED_API_KEY',
                'NEWS_API_KEY',
                'SystemConfig'
            ]
            for field in required_fields:
                self.assertTrue(hasattr(config, field), f"Missing required config field: {field}")
        except ImportError:
            self.skipTest("config.py not available")
    
    def test_module_imports(self):
        """Test if all necessary modules can be imported"""
        modules = [
            'modules.data_acquisition',
            'modules.asset_selector',
            'modules.ml_predictor',
            'modules.sentiment_analysis',
            'modules.signal_aggregator',
            'modules.risk_manager',
            'modules.broker_integration',
            'modules.backtest'
        ]
        
        for module in modules:
            try:
                importlib.import_module(module)
                self.assertTrue(True)
            except ImportError as e:
                self.fail(f"Failed to import {module}: {str(e)}")
    
    def test_directory_structure(self):
        """Test if required directories exist"""
        required_dirs = ['data', 'models', 'logs', 'cache', 'results', 'reports', 'plots']
        for dir_name in required_dirs:
            self.assertTrue(os.path.isdir(dir_name) or os.access(os.path.dirname(dir_name), os.W_OK), 
                           f"Directory {dir_name} doesn't exist or is not writable")