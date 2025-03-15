import streamlit as st
import pandas as pd
import logging

st.title("📈 Real-Time Trading Dashboard")

# Function to safely load trading data
@st.cache_data
def load_trading_data():
    try:
        data = pd.read_csv('data/historical_prices.csv')
        logging.info("Trading data loaded successfully.")
    except Exception as e:
        logging.warning(f"Error loading data: {e}")
        data = pd.DataFrame({
            'Date': pd.date_range('2024-01-01', periods=10),
            'Close': [40000, 40500, 41000, 41500, 42000, 42500, 43000, 43500, 44000, 44500]
        })
    return data

# Load data
data = load_trading_data()

# Ensure date column is correctly named
date_column = None
for col in ['Date', 'date', 'timestamp', 'Timestamp']:
    if col in data.columns:
        date_column = col
        break

if date_column is None:
    st.error("No date or timestamp column found in data.")
else:
    data[date_column] = pd.to_datetime(data[date_column])

    st.subheader("📊 Historical Closing Prices")
    st.line_chart(data.set_index(date_column)['Close'])

    st.subheader("🗃️ Raw Data")
    st.dataframe(data)

# Display recent logs
st.subheader("📌 Recent Logs")
try:
    with open('trading.log', 'r') as log_file:
        log_entries = log_file.readlines()[-10:]  # Display last 10 logs
    st.text("Recent Logs:")
    for entry in reversed(log_entries):
        st.write(entry.strip())
except Exception as e:
    st.warning(f"Unable to load logs: {e}")
