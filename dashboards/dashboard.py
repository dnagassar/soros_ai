# dashboards/dashboard.py
import streamlit as st
import pandas as pd
import time

def load_trading_data():
    try:
        data = pd.read_csv('data/trading_performance.csv')
    except Exception:
        data = pd.DataFrame({
            'timestamp': pd.date_range(start='2021-01-01', periods=10, freq='T'),
            'PnL': [100, 105, 102, 110, 108, 115, 112, 117, 120, 118]
        })
    return data

st.title("Real-Time Trading Dashboard")

data = load_trading_data()
st.dataframe(data)
st.line_chart(data.set_index('timestamp')['PnL'])

# Auto-refresh every 10 seconds
st.experimental_singleton.clear()  # Optionally clear cache
time.sleep(10)
st.experimental_rerun()
