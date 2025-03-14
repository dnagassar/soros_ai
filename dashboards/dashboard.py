# dashboards/dashboard.py
import streamlit as st
import pandas as pd
import time

st.title("Real-Time Trading Dashboard")

def load_trading_logs():
    try:
        return pd.read_csv('trading.log', sep='\n', header=None, names=['Logs'])
    except:
        return pd.DataFrame({'Logs': ["No logs available yet."]})

data = load_trading_data()
logs = load_trading_logs()

st.subheader("Profit and Loss (PnL)")
st.line_chart(data.set_index('timestamp')['PnL'])

st.subheader("Live Trading Logs")
st.dataframe(logs)

# Auto-refresh dashboard every minute
st.experimental_rerun()
