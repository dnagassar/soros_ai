# modules/strategy.py
import alpaca_trade_api as tradeapi
import logging  # Added missing import
import sys
from config import ALPACA_API_KEY, ALPACA_SECRET_KEY, ALPACA_BASE_URL

# Ensure Alpaca API keys are properly set.
if ALPACA_API_KEY in [None, 'YOUR_ALPACA_API_KEY'] or ALPACA_SECRET_KEY in [None, 'YOUR_ALPACA_SECRET_KEY']:
    sys.exit("Error: Alpaca API keys are not set. Please update config.py or set the environment variables ALPACA_API_KEY and ALPACA_SECRET_KEY.")

api = tradeapi.REST(ALPACA_API_KEY, ALPACA_SECRET_KEY, ALPACA_BASE_URL)

def execute_trade(symbol, qty, side='buy'):
    try:
        order = api.submit_order(
            symbol=symbol,
            qty=qty,
            side=side,
            type='market',
            time_in_force='day'
        )
        logging.info(f"{side.capitalize()} order for {qty} shares of {symbol} executed successfully. Order ID: {order.id}")
    except Exception as e:
        logging.error(f"Order execution error for {symbol}: {e}")
