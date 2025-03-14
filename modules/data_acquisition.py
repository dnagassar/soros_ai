import yfinance as yf
from alpha_vantage.timeseries import TimeSeries
import pandas as pd
from config import ALPHA_VANTAGE_API_KEY
import logging

def fetch_price_data(ticker, start, end):
    try:
        data = yf.download(ticker, start=start, end=end)
        if data.empty:
            raise ValueError("Empty data from yfinance.")
        logging.info("Fetched data from yfinance.")
    except Exception as e:
        logging.warning(f"yfinance error: {e}. Falling back to Alpha Vantage.")
        ts = TimeSeries(key=ALPHA_VANTAGE_API_KEY, output_format='pandas')
        data, _ = ts.get_daily(symbol=ticker, outputsize='full')
        data = data.loc[start:end]
    return data
