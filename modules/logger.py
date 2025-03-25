# modules/logger.py
import logging
import os
from datetime import datetime

def setup_logger(name=None, level='INFO', log_file=None):
    """
    Set up a logger with the specified name, level, and optional file output
    
    Parameters:
      - name: Logger name (None for root logger)
      - level: Logging level ('DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL')
      - log_file: Optional log file path
      
    Returns:
      - Logger instance
    """
    # Convert string level to logging constant
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(numeric_level)
    
    # Remove existing handlers to prevent duplicates
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    
    # Add console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Add file handler if requested
    if log_file:
        # Create log directory if needed
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger