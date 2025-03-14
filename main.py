# main.py
import datetime
import numpy as np
import pandas as pd
import backtrader as bt

# Import custom modules
from modules.data_acquisition import fetch_price_data
from modules.signal_aggregator import aggregate_signals
from modules.logger import log_trade
from modules.strategy import SentimentStrategy

# =====================================================
# STEP 1: DATA ACQUISITION & CSV CLEANING
# =====================================================

# Fetch price data for AAPL between specified dates
data_df = fetch_price_data('AAPL', '2020-01-01', '2020-12-31')

# Reset index so that the index becomes a column
data_df.reset_index(inplace=True)

# If columns are a MultiIndex, flatten them
if isinstance(data_df.columns, pd.MultiIndex):
    # Use only the first level of the MultiIndex
    data_df.columns = [col[0] for col in data_df.columns.values]

print("Columns after flattening:", data_df.columns.tolist())

# Ensure that the date column is named 'Date'
if 'Date' not in data_df.columns:
    # Rename the first column (if it looks like a date)
    first_col = data_df.columns[0]
    data_df.rename(columns={first_col: 'Date'}, inplace=True)

# Insert a "Ticker" column at the beginning if not present
if 'Ticker' not in data_df.columns:
    data_df.insert(0, 'Ticker', 'AAPL')

# Reorder columns to the desired order:
# We want: Ticker, Date, Open, High, Low, Close, Volume
# yfinance returns: Date, Close, High, Low, Open, Volume (after flattening)
# After inserting Ticker, the order becomes: Ticker, Date, Close, High, Low, Open, Volume.
desired_order = ['Ticker', 'Date', 'Open', 'High', 'Low', 'Close', 'Volume']
# Only reorder if all desired columns are present:
if set(desired_order).issubset(data_df.columns):
    data_df = data_df[desired_order]
else:
    print("Warning: Not all desired columns are present. Current columns:", data_df.columns.tolist())

# Clean the 'Date' column: convert to string, strip whitespace,
# and remove rows with an empty or NaN 'Date'
data_df['Date'] = data_df['Date'].astype(str).str.strip()
data_df = data_df[data_df['Date'] != '']
data_df.dropna(subset=['Date'], inplace=True)

# Save the cleaned CSV file
clean_csv_path = 'data/historical_prices_clean.csv'
data_df.to_csv(clean_csv_path, index=False)
print("Clean CSV saved to", clean_csv_path)

# =====================================================
# STEP 2: SIGNAL AGGREGATION (Example using dummy data)
# =====================================================

news_text = "Company reports record earnings, bullish sentiment prevails."
technical_signal = 1
symbol = "AAPL"
X_train = np.random.rand(100, 10)  # Dummy features for ML models
y_train = np.random.rand(100)      # Dummy target values
X_test = np.random.rand(1, 10)      # Dummy test features

final_signal = aggregate_signals(news_text, technical_signal, symbol, X_train, y_train, X_test)
print("Final Aggregated Signal:", final_signal)
log_trade(f"Aggregated Signal for {symbol}: {final_signal}")

# =====================================================
# STEP 3: SETUP BACKTRADER & RUN STRATEGY
# =====================================================

cerebro = bt.Cerebro()
cerebro.addstrategy(SentimentStrategy)

# Create a data feed using GenericCSVData with proper column mapping.
# Our cleaned CSV is assumed to have columns:
# Ticker,Date,Open,High,Low,Close,Volume
data = bt.feeds.GenericCSVData(
    dataname=clean_csv_path,
    dtformat='%Y-%m-%d',
    datetime=1,   # 'Date' column is at index 1 (Ticker is at index 0)
    open=2,       # Open price at index 2
    high=3,       # High price at index 3
    low=4,        # Low price at index 4
    close=5,      # Close price at index 5
    volume=6,     # Volume at index 6
    openinterest=-1,
    headers=True,
    nullvalue=0.0,
    fromdate=datetime.datetime(2020, 1, 1),
    todate=datetime.datetime(2020, 12, 31)
)

cerebro.adddata(data)
cerebro.run()
cerebro.plot()
