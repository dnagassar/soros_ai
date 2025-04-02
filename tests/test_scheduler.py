import unittest
import sys
import os
import importlib.util

class TestScheduler(unittest.TestCase):
    
    def test_scheduler_imports(self):
        """Test if scheduler.py can be imported without errors"""
        try:
            spec = importlib.util.spec_from_file_location("scheduler", "scheduler.py")
            scheduler = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(scheduler)
            self.assertTrue(True)
        except Exception as e:
            self.fail(f"Failed to import scheduler.py: {str(e)}")
    
    def test_scheduler_functions(self):
        """Test if key functions in scheduler can be called"""
        try:
            # Import the scheduler
            spec = importlib.util.spec_from_file_location("scheduler", "scheduler.py")
            scheduler = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(scheduler)
            
            # Check if key functions exist
            self.assertTrue(hasattr(scheduler, "initialize_system"))
            
            # Try to initialize the system
            scheduler.initialize_system()
            self.assertTrue(True)
        except Exception as e:
            self.fail(f"Failed to call scheduler functions: {str(e)}")