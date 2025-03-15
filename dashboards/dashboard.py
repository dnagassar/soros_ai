# dashboards/dashboard.py
import sys
import os

# Add the parent directory to the sys.path so that modules can be found.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
import pandas as pd
import time
from modules.strategy_insights import get_strategy_insights

def load_trading_data():
    """
    Loads trading performance data from a CSV file.
    If the file does not exist, returns a dummy DataFrame.
    """
    try:
        data = pd.read_csv('data/trading_performance.csv')
    except Exception:
        data = pd.DataFrame({
            'timestamp': pd.date_range(start='2021-01-01', periods=10, freq='min'),
            'PnL': [100, 105, 102, 110, 108, 115, 112, 117, 120, 118]
        })
    return data

st.title("Real-Time Trading Dashboard")

# Display trading performance data
data = load_trading_data()
st.dataframe(data)
st.line_chart(data.set_index('timestamp')['PnL'])

# Prepare performance metrics (use real data in production)
performance_data = {
    "PnL": "$500",
    "signal_accuracy": "75%",
    "drawdown": "2%",
    "number_of_trades": "25",
    "avg_position_size": "100 shares"
}

# Get LLM-generated strategy insights
st.subheader("Strategy Insights")
insights = get_strategy_insights(performance_data)
st.write(insights)

# Auto-refresh the dashboard every 10 seconds
time.sleep(10)
st.experimental_rerun()
