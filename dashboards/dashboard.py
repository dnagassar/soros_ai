import sys
import os
# Ensure the project root is in sys.path so config.py can be imported
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import openai
from config import OPENAI_API_KEY

# Set OpenAI API key
openai.api_key = OPENAI_API_KEY

@st.cache_data
def load_trading_data():
    """
    Load live trading performance data from CSV.
    If unavailable, generate dummy data.
    """
    try:
        data = pd.read_csv('data/trading_performance.csv', parse_dates=['timestamp'])
    except Exception:
        dates = pd.date_range(start='2021-01-01 09:30', periods=50, freq='T')
        pnl = np.random.normal(0, 5, size=50)
        data = pd.DataFrame({'timestamp': dates, 'PnL': pnl})
    return data

@st.cache_data
def load_backtest_data():
    """
    Load backtest performance data from CSV.
    If unavailable, generate dummy backtest data.
    """
    try:
        data = pd.read_csv('data/backtest_results.csv', parse_dates=['timestamp'])
    except Exception:
        dates = pd.date_range(start='2022-01-01', periods=100, freq='D')
        pnl = np.random.normal(0, 10, size=100)
        data = pd.DataFrame({'timestamp': dates, 'PnL': pnl})
    return data

def compute_strategy_metrics(df):
    """
    Computes key performance metrics:
    - Cumulative Equity (sum of PnL)
    - Total number of trades
    - Total PnL
    - Average trade PnL
    - Standard deviation of trade PnL
    - Maximum drawdown
    - Simplified Sharpe ratio (average PnL / std deviation)
    """
    df = df.copy()
    df['Equity'] = df['PnL'].cumsum()
    
    total_trades = len(df)
    total_pnl = df['Equity'].iloc[-1]
    avg_trade_pnl = df['PnL'].mean()
    std_trade_pnl = df['PnL'].std()
    
    running_max = df['Equity'].cummax()
    drawdown = (df['Equity'] - running_max) / running_max
    max_drawdown = drawdown.min()
    
    sharpe_ratio = avg_trade_pnl / std_trade_pnl if std_trade_pnl != 0 else np.nan
    
    metrics = {
        'Total Trades': total_trades,
        'Total PnL': total_pnl,
        'Average Trade PnL': avg_trade_pnl,
        'Std Trade PnL': std_trade_pnl,
        'Max Drawdown': max_drawdown,
        'Sharpe Ratio': sharpe_ratio
    }
    return df, metrics

def generate_strategy_overview(live_metrics, backtest_metrics):
    """
    Combines live and backtest performance metrics and sends them to OpenAI's LLM
    to generate a professional strategy overview and recommendations.
    """
    # Format metrics into text blocks
    live_text = "\n".join([f"{key}: {value:.2f}" for key, value in live_metrics.items()])
    backtest_text = "\n".join([f"{key}: {value:.2f}" for key, value in backtest_metrics.items()])
    
    combined_prompt = (
        "Below are the performance metrics for our live trading data:\n"
        f"{live_text}\n\n"
        "And below are the metrics for our backtested results:\n"
        f"{backtest_text}\n\n"
        "Please provide a professional analysis of our trading strategy. "
        "Identify strengths and weaknesses, and recommend which parameters (e.g., adaptive position sizing, "
        "signal decay, stop-loss levels, etc.) should be fine-tuned to improve overall performance. "
        "The response should be detailed, actionable, and focused on how to refine the system to consistently outperform market benchmarks."
    )
    
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a seasoned financial quantitative analyst."},
                {"role": "user", "content": combined_prompt}
            ],
            temperature=0.7,
            max_tokens=300
        )
        overview = response.choices[0].message.content
    except Exception as e:
        overview = f"Error generating strategy overview: {e}"
    
    return overview

st.title("Strategy Performance & Refinement Insights Dashboard")

# Load live and backtest data
live_data = load_trading_data()
backtest_data = load_backtest_data()

# Compute metrics
live_data, live_metrics = compute_strategy_metrics(live_data)
backtest_data, backtest_metrics = compute_strategy_metrics(backtest_data)

# Display live performance metrics
st.subheader("Live Trading Performance Metrics")
for key, value in live_metrics.items():
    st.write(f"**{key}:** {value:.2f}")

# Display backtest performance metrics
st.subheader("Backtest Performance Metrics")
for key, value in backtest_metrics.items():
    st.write(f"**{key}:** {value:.2f}")

# Generate strategy overview using combined metrics
with st.spinner("Generating professional strategy overview..."):
    overview = generate_strategy_overview(live_metrics, backtest_metrics)

st.subheader("Professional Strategy Overview & Recommendations")
st.write(overview)

# Visualizations
st.subheader("Live Trading Equity Curve")
st.line_chart(live_data.set_index('timestamp')['Equity'])

st.subheader("Backtest Equity Curve")
st.line_chart(backtest_data.set_index('timestamp')['Equity'])

st.subheader("Live PnL Distribution")
live_pnl_hist = live_data['PnL'].round(1).value_counts().sort_index()
st.bar_chart(live_pnl_hist)

st.subheader("Backtest PnL Distribution")
backtest_pnl_hist = backtest_data['PnL'].round(1).value_counts().sort_index()
st.bar_chart(backtest_pnl_hist)

st.subheader("Custom Matplotlib Plot - Combined Equity Curves")
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(live_data['timestamp'], live_data['Equity'], label='Live Equity Curve')
ax.plot(backtest_data['timestamp'], backtest_data['Equity'], label='Backtest Equity Curve', linestyle='--')
ax.set_xlabel('Timestamp')
ax.set_ylabel('Equity')
ax.set_title('Live vs Backtest Equity Curves')
ax.legend()
st.pyplot(fig)
