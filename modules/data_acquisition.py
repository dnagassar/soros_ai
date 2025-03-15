import yfinance as yf
from alpha_vantage.timeseries import TimeSeries
import pandas as pd
import logging
from config import ALPHA_VANTAGE_API_KEY

def fetch_price_data(ticker, start, end, asset_type='stock'):
    try:
        if asset_type == 'crypto':
            ticker_formatted = f"{ticker}-USD"
        else:
            ticker = ticker
        data = yf.download(ticker, start=start, end=end)
        if data.empty:
            raise ValueError("Empty data from yfinance.")
        logging.info("Data fetched successfully from yfinance.")
    except Exception as e:
        logging.warning(f"yfinance error: {e}. Using Alpha Vantage as fallback.")
        ts = TimeSeries(key=ALPHA_VANTAGE_API_KEY, output_format='pandas')
        symbol = ticker if asset_type == 'stock' else f"{ticker}USD"
        data, _ = ts.get_daily(symbol=symbol, outputsize='full')
        data = data.loc[start:end]
    return data
