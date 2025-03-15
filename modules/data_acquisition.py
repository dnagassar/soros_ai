# modules/data_acquisition.py
import yfinance as yf
from alpha_vantage.timeseries import TimeSeries
import pandas as pd
from config import ALPHA_VANTAGE_API_KEY  # Ensure your API key is stored in config.py

def fetch_price_data(ticker, start, end):
    try:
        # Primary: using yfinance
        data = yf.download(ticker, start=start, end=end)
        if data.empty:
            raise ValueError("No data from yfinance.")
        print("Data fetched from yfinance")
    except Exception as e:
        print("Falling back to Alpha Vantage:", e)
        ts = TimeSeries(key=ALPHA_VANTAGE_API_KEY, output_format='pandas')
        data, _ = ts.get_daily(symbol=ticker, outputsize='full')
        data = data.loc[start:end]
    
    # If the columns are a MultiIndex, flatten them and force them to strings
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = [str(col[0]) for col in data.columns]
    else:
        data.columns = [str(col) for col in data.columns]
    return data

if __name__ == "__main__":
    df = fetch_price_data('AAPL', '2020-01-01', '2020-12-31')
    df.to_csv('data/historical_prices.csv')
