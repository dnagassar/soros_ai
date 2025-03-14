import logging

logging.basicConfig(
    filename='trading.log', level=logging.INFO,
    format='%(asctime)s %(levelname)s: %(message)s'
)

def log_event(message, level='info'):
    getattr(logging, level)(message)
