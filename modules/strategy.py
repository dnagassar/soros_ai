import alpaca_trade_api as tradeapi
import logging
import sys
from config import ALPACA_API_KEY, ALPACA_SECRET_KEY, ALPACA_BASE_URL

if ALPACA_API_KEY in [None, 'YOUR_ALPACA_API_KEY'] or ALPACA_SECRET_KEY in [None, 'YOUR_ALPACA_SECRET_KEY']:
    sys.exit("Error: Alpaca API keys are not set. Please update config.py or set environment variables.")

api = tradeapi.REST(ALPACA_API_KEY, ALPACA_SECRET_KEY, ALPACA_BASE_URL)

def execute_trade(symbol, qty, side='buy'):
    try:
        order = api.submit_order(
            symbol=symbol,
            qty=qty,
            side=side,
            type='market',
            time_in_force='gtc'  # Good until canceled is common in crypto
        )
        logging.info(f"{side.capitalize()} order for {qty} {symbol} executed successfully. Order ID: {order.id}")
    except Exception as e:
        logging.error(f"Order execution error for {symbol}: {e}")
