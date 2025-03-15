# modules/logger.py
import logging

logging.basicConfig(
    filename='trading.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def log_trade(trade_info):
    logging.info(trade_info)

if __name__ == "__main__":
    log_trade("Test trade executed: BUY 100 AAPL at $150")
