# dashboards/dashboard.py
import sys
import os
# Ensure the project root is in sys.path so config.py can be imported
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf
import json
from datetime import datetime, timedelta
import logging
from io import StringIO
import altair as alt
import hashlib
import openai
import time
from config import OPENAI_API_KEY, DASHBOARD_PASSWORD

# Set OpenAI API key
openai.api_key = OPENAI_API_KEY

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('dashboard.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Create cache directory
os.makedirs('cache', exist_ok=True)

# Authentication function
def check_password():
    """Returns `True` if the user had the correct password."""
    
    if "password_correct" not in st.session_state:
        # First run, show input for password
        st.session_state.password_correct = False
        
    if st.session_state.password_correct:
        return True
        
    # Password input
    password = st.text_input("Enter dashboard password", type="password")
    
    # Check if password is correct
    password_hash = hashlib.sha256(password.encode()).hexdigest()
    if password_hash == DASHBOARD_PASSWORD:
        st.session_state.password_correct = True
        return True
    else:
        if password:
            st.error("Incorrect password")
        return False

@st.cache_data(ttl=300)  # Cache with 5-minute TTL
def load_trading_data(file_path='data/trading_performance.csv'):
    """
    Load live trading performance data from CSV.
    If unavailable, generate dummy data.
    """
    try:
        data = pd.read_csv(file_path, parse_dates=['timestamp'])
        logger.info(f"Loaded trading data from {file_path} with {len(data)} rows")
        return data
    except Exception as e:
        logger.warning(f"Error loading trading data: {e}. Generating dummy data.")
        # Generate more realistic dummy data
        np.random.seed(42)  # For reproducibility
        
        # Create dates for the past 50 days
        end_date = datetime.now()
        start_date = end_date - timedelta(days=50)
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        
        # Create a trend with some noise
        trend = np.linspace(0, 10, len(dates))
        noise = np.random.normal(0, 2, size=len(dates))
        cumulative_pnl = trend + noise.cumsum()
        
        # Daily PnL
        daily_pnl = np.diff(cumulative_pnl, prepend=0)
        
        # Create a DataFrame
        data = pd.DataFrame({
            'timestamp': dates,
            'PnL': daily_pnl,
            'Equity': cumulative_pnl + 100000,  # Start with $100k
            'Trades': np.random.randint(1, 10, size=len(dates)),
            'Win_Rate': np.random.uniform(0.4, 0.7, size=len(dates)) * 100,
            'Drawdown': np.random.uniform(0, 0.05, size=len(dates)) * 100,
            'Sharpe': np.random.uniform(0.5, 2.0, size=len(dates)),
            'Market_Return': np.random.normal(0.0005, 0.01, size=len(dates)) * 100
        })
        
        return data

@st.cache_data(ttl=600)  # Cache with 10-minute TTL
def load_backtest_data(file_path='results/backtest_results.json'):
    """
    Load backtest performance data from JSON.
    If unavailable, generate dummy backtest data.
    """
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        logger.info(f"Loaded backtest data from {file_path}")
        return data
    except Exception as e:
        logger.warning(f"Error loading backtest data: {e}. Generating dummy data.")
        
        # Generate dummy backtest results
        np.random.seed(84)  # Different seed for variety
        
        metrics = {
            "final_value": 115000,
            "pnl": 15000,
            "pnl_percent": 15.0,
            "sharpe": 1.2,
            "max_drawdown": 8.5,
            "max_drawdown_length": 15,
            "sqn": 1.8,
            "vwr": 0.95,
            "win_rate": 58.5,
            "total_trades": 125,
            "avg_trade_pnl": 120.0,
            "avg_winning_trade": 350.0,
            "avg_losing_trade": -180.0,
            "largest_winning_trade": 1200.0,
            "largest_losing_trade": -800.0
        }
        
        benchmark = {
            "symbol": "^GSPC",
            "start_price": 4000.0,
            "end_price": 4400.0,
            "return_percent": 10.0
        }
        
        metadata = {
            "strategy": {
                "name": "AdaptiveSentimentStrategy",
                "parameters": {
                    "sentiment_period": 3,
                    "vol_window": 20,
                    "ema_short": 10,
                    "ema_medium": 30,
                    "ema_long": 50,
                    "rsi_period": 14,
                    "stop_loss": 0.03,
                    "take_profit": 0.05,
                    "risk_factor": 0.01,
                    "ml_weight": 0.4,
                    "sentiment_weight": 0.3,
                    "tech_weight": 0.3
                }
            },
            "data": {
                "symbols": ["AAPL", "MSFT", "GOOGL", "AMZN", "META"],
                "start_date": "2022-01-01",
                "end_date": "2022-12-31"
            }
        }
        
        # Create dummy daily equity curve
        dates = pd.date_range(
            start=pd.to_datetime(metadata["data"]["start_date"]),
            end=pd.to_datetime(metadata["data"]["end_date"]),
            freq='D'
        )
        
        trend = np.linspace(0, metrics["pnl"], len(dates))
        noise = np.random.normal(0, metrics["pnl"] * 0.01, size=len(dates))
        equity_curve = 100000 + trend + noise.cumsum()
        
        # Daily PnL
        daily_pnl = np.diff(equity_curve, prepend=equity_curve[0])
        
        # Create a daily performance DataFrame
        daily_data = []
        for i, date in enumerate(dates):
            daily_data.append({
                "date": date.strftime('%Y-%m-%d'),
                "equity": equity_curve[i],
                "pnl": daily_pnl[i],
                "drawdown": max(0, (max(equity_curve[:i+1]) - equity_curve[i]) / max(equity_curve[:i+1]) * 100) if i > 0 else 0
            })
        
        # Dummy trade list
        trades = []
        current_date = pd.to_datetime(metadata["data"]["start_date"])
        end_date = pd.to_datetime(metadata["data"]["end_date"])
        
        while current_date < end_date and len(trades) < metrics["total_trades"]:
            # Random trade details
            symbol = np.random.choice(metadata["data"]["symbols"])
            is_win = np.random.random() < (metrics["win_rate"] / 100)
            
            # Generate trade outcomes based on average win/loss
            if is_win:
                pnl = np.random.normal(metrics["avg_winning_trade"], metrics["avg_winning_trade"] * 0.3)
            else:
                pnl = np.random.normal(metrics["avg_losing_trade"], abs(metrics["avg_losing_trade"]) * 0.3)
            
            # Random hold period
            hold_days = np.random.randint(1, 10)
            entry_date = current_date
            exit_date = entry_date + timedelta(days=hold_days)
            
            trades.append({
                "symbol": symbol,
                "entry_date": entry_date.strftime('%Y-%m-%d'),
                "exit_date": exit_date.strftime('%Y-%m-%d'),
                "entry_price": np.random.uniform(50, 200),
                "exit_price": 0,  # Will calculate based on PnL
                "size": np.random.randint(10, 100),
                "pnl": pnl,
                "pnl_percent": np.random.uniform(-5, 8)
            })
            
            # Calculate exit price based on entry, size and PnL
            trades[-1]["exit_price"] = trades[-1]["entry_price"] + (trades[-1]["pnl"] / trades[-1]["size"])
            
            # Advance date by a random amount
            current_date += timedelta(days=np.random.randint(1, 5))
        
        # Combine everything
        dummy_backtest = {
            "metrics": metrics,
            "benchmark": benchmark,
            "metadata": metadata,
            "daily_performance": daily_data,
            "trades": trades
        }
        
        return dummy_backtest

@st.cache_data(ttl=3600)  # Cache with 1-hour TTL
def load_market_data(symbols, period='6mo'):
    """Load market data for given symbols"""
    try:
        data = {}
        
        for symbol in symbols:
            ticker_data = yf.download(symbol, period=period)
            if not ticker_data.empty:
                data[symbol] = ticker_data
        
        logger.info(f"Loaded market data for {len(data)} symbols")
        return data
    except Exception as e:
        logger.error(f"Error loading market data: {e}")
        return {}

@st.cache_data(ttl=600)
def load_log_file(log_file='trading.log', max_lines=100):
    """Load the trading log file"""
    try:
        with open(log_file, 'r') as f:
            # Get the last max_lines lines
            lines = f.readlines()[-max_lines:]
        
        logger.info(f"Loaded {len(lines)} lines from log file {log_file}")
        return lines
    except Exception as e:
        logger.warning(f"Error loading log file: {e}")
        return []

def parse_log_entries(log_lines):
    """Parse log entries into structured format"""
    entries = []
    
    for line in log_lines:
        try:
            # Common format: timestamp - level - message
            parts = line.strip().split(' - ', 2)
            
            if len(parts) >= 3:
                timestamp = parts[0]
                level = parts[1]
                message = parts[2]
                
                entries.append({
                    'timestamp': pd.to_datetime(timestamp),
                    'level': level,
                    'message': message
                })
        except Exception as e:
            logger.error(f"Error parsing log line: {e}")
    
    return pd.DataFrame(entries)

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
    
    if 'Equity' not in df.columns:
        df['Equity'] = df['PnL'].cumsum()
    
    total_trades = df['Trades'].sum() if 'Trades' in df.columns else len(df)
    total_pnl = df['PnL'].sum()
    avg_trade_pnl = total_pnl / max(1, total_trades)
    std_trade_pnl = df['PnL'].std()
    
    running_max = df['Equity'].cummax()
    drawdown = (df['Equity'] - running_max) / running_max
    max_drawdown = drawdown.min() * 100  # Convert to percentage
    
    if 'Sharpe' in df.columns:
        sharpe_ratio = df['Sharpe'].mean()
    else:
        sharpe_ratio = avg_trade_pnl / std_trade_pnl if std_trade_pnl != 0 else np.nan
    
    # Calculate win rate if available
    if 'Win_Rate' in df.columns:
        win_rate = df['Win_Rate'].mean()
    else:
        win_rate = np.nan
        
    # Calculate additional risk metrics
    volatility = df['PnL'].std() / df['Equity'].mean() * 100  # Daily volatility as percentage
    best_day = df['PnL'].max()
    worst_day = df['PnL'].min()
    
    # Calculate historical correlation with market
    if 'Market_Return' in df.columns:
        market_correlation = df['PnL'].corr(df['Market_Return'])
    else:
        market_correlation = np.nan
    
    metrics = {
        'Total Trades': total_trades,
        'Total PnL': total_pnl,
        'Final Equity': df['Equity'].iloc[-1],
        'Return (%)': (df['Equity'].iloc[-1] / df['Equity'].iloc[0] - 1) * 100,
        'Average Trade PnL': avg_trade_pnl,
        'Win Rate (%)': win_rate,
        'Std Trade PnL': std_trade_pnl,
        'Max Drawdown (%)': max_drawdown,
        'Sharpe Ratio': sharpe_ratio,
        'Volatility (%)': volatility,
        'Best Day': best_day,
        'Worst Day': worst_day,
        'Market Correlation': market_correlation
    }
    
    return df, metrics

def generate_ai_insights(live_metrics, backtest_metrics, market_data=None):
    """
    Generates AI-powered trading strategy insights using the LLM.
    """
    # Prepare input data
    live_text = "\n".join([f"{key}: {value:.2f}" for key, value in live_metrics.items()])
    backtest_text = json.dumps(backtest_metrics)
    
    # Add market data summary if available
    market_summary = ""
    if market_data:
        for symbol, data in market_data.items():
            if not data.empty:
                returns = (data['Close'].iloc[-1] / data['Close'].iloc[0] - 1) * 100
                volatility = data['Close'].pct_change().std() * 100
                market_summary += f"{symbol}: Return={returns:.2f}%, Volatility={volatility:.2f}%\n"
    
    # Create prompt
    prompt = f"""
    You are a professional quantitative trading analyst. Based on the following metrics, provide a concise analysis of our trading strategy's performance, highlighting key strengths and weaknesses, and suggesting specific improvements.
    
    LIVE TRADING METRICS:
    {live_text}
    
    BACKTEST METRICS:
    {backtest_text}
    
    MARKET DATA SUMMARY:
    {market_summary}
    
    Your analysis should cover:
    1. Key performance observations (max 3 bullet points)
    2. Potential issues or risks (max 2 bullet points)
    3. Specific, actionable recommendations for improvement (max 3 bullet points)
    
    Keep your entire response under 300 words, focused on the most important insights only.
    """
    
    # Cache key for this analysis
    cache_key = hashlib.md5(prompt.encode()).hexdigest()
    cache_file = f"cache/ai_insights_{cache_key}.txt"
    
    # Check if we have a cached result
    if os.path.exists(cache_file) and (time.time() - os.path.getmtime(cache_file) < 3600):  # 1-hour cache
        try:
            with open(cache_file, 'r') as f:
                return f.read()
        except Exception:
            pass  # If reading fails, generate a new analysis
    
    # Generate insights using the LLM
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",  # Use the best available model for quality insights
            messages=[
                {"role": "system", "content": "You are a professional quantitative trading analyst with expertise in algorithmic trading strategies."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.5,  # Lower temperature for more consistent, factual responses
            max_tokens=400
        )
        
        insights = response.choices[0].message.content.strip()
        
        # Cache the result
        try:
            with open(cache_file, 'w') as f:
                f.write(insights)
        except Exception as e:
            logger.error(f"Error caching AI insights: {e}")
        
        return insights
    
    except Exception as e:
        logger.error(f"Error generating AI insights: {e}")
        return f"Error generating strategy insights: {str(e)}"

def plot_equity_curve(df):
    """Create an interactive equity curve plot"""
    fig = go.Figure()
    
    # Add equity curve
    fig.add_trace(go.Scatter(
        x=df['timestamp'],
        y=df['Equity'],
        mode='lines',
        name='Equity',
        line=dict(color='blue', width=2)
    ))
    
    # Calculate and add drawdown
    max_equity = df['Equity'].cummax()
    drawdown = (df['Equity'] - max_equity) / max_equity * 100
    
    fig.add_trace(go.Scatter(
        x=df['timestamp'],
        y=drawdown,
        mode='lines',
        name='Drawdown %',
        line=dict(color='red', width=1, dash='dot'),
        yaxis='y2'
    ))
    
    # Update layout with two y-axes
    fig.update_layout(
        title='Equity Curve and Drawdown',
        xaxis_title='Date',
        yaxis_title='Equity ($)',
        yaxis2=dict(
            title='Drawdown (%)',
            overlaying='y',
            side='right',
            range=[min(drawdown) * 1.5, 1],  # Adjust range for better visualization
            showgrid=False
        ),
        height=500,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    return fig

def plot_pnl_distribution(df):
    """Create a histogram of daily PnL"""
    fig = go.Figure()
    
    fig.add_trace(go.Histogram(
        x=df['PnL'],
        nbinsx=30,
        marker_color='blue',
        opacity=0.7
    ))
    
    # Add mean line
    mean_pnl = df['PnL'].mean()
    fig.add_vline(x=mean_pnl, line_dash="dash", line_color="green", 
                 annotation_text=f"Mean: ${mean_pnl:.2f}", 
                 annotation_position="top right")
    
    # Add zero line
    fig.add_vline(x=0, line_dash="solid", line_color="red")
    
    fig.update_layout(
        title='Daily P&L Distribution',
        xaxis_title='P&L ($)',
        yaxis_title='Frequency',
        height=400
    )
    
    return fig

def plot_win_rate_over_time(df):
    """Create a line chart of win rate over time"""
    if 'Win_Rate' not in df.columns:
        return None
    
    # Create a smoothed win rate using rolling average
    df_smoothed = df.copy()
    df_smoothed['Win_Rate_Smoothed'] = df['Win_Rate'].rolling(window=7, min_periods=1).mean()
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=df_smoothed['timestamp'],
        y=df_smoothed['Win_Rate'],
        mode='markers',
        name='Daily Win Rate',
        marker=dict(size=4, color='blue', opacity=0.5)
    ))
    
    fig.add_trace(go.Scatter(
        x=df_smoothed['timestamp'],
        y=df_smoothed['Win_Rate_Smoothed'],
        mode='lines',
        name='7-Day Average',
        line=dict(color='green', width=2)
    ))
    
    fig.update_layout(
        title='Win Rate Over Time',
        xaxis_title='Date',
        yaxis_title='Win Rate (%)',
        height=400,
        yaxis=dict(range=[0, 100])
    )
    
    return fig

def plot_trading_activity(df):
    """Create a bar chart of trading activity over time"""
    if 'Trades' not in df.columns:
        return None
    
    # Resample to weekly
    weekly_trades = df.resample('W', on='timestamp')['Trades'].sum().reset_index()
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=weekly_trades['timestamp'],
        y=weekly_trades['Trades'],
        marker_color='blue',
        opacity=0.7
    ))
    
    fig.update_layout(
        title='Weekly Trading Activity',
        xaxis_title='Week',
        yaxis_title='Number of Trades',
        height=400
    )
    
    return fig

def plot_backtest_trades(trades_data):
    """Create a scatter plot of trades from backtest"""
    if not trades_data:
        return None
    
    # Convert to DataFrame
    trades_df = pd.DataFrame(trades_data)
    
    # Convert date strings to datetime
    trades_df['entry_date'] = pd.to_datetime(trades_df['entry_date'])
    trades_df['exit_date'] = pd.to_datetime(trades_df['exit_date'])
    
    # Calculate trade duration
    trades_df['duration'] = (trades_df['exit_date'] - trades_df['entry_date']).dt.days
    
    # Create color map based on PnL
    trades_df['color'] = np.where(trades_df['pnl'] >= 0, 'green', 'red')
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=trades_df['exit_date'],
        y=trades_df['pnl'],
        mode='markers',
        marker=dict(
            size=trades_df['pnl'].abs() / trades_df['pnl'].abs().max() * 15 + 5,
            color=trades_df['color'],
            opacity=0.7,
            line=dict(width=1, color='black')
        ),
        text=trades_df.apply(lambda row: f"Symbol: {row['symbol']}<br>Entry: ${row['entry_price']:.2f}<br>Exit: ${row['exit_price']:.2f}<br>PnL: ${row['pnl']:.2f}<br>Duration: {row['duration']} days", axis=1),
        hoverinfo='text',
        name='Trades'
    ))
    
    # Add zero line
    fig.add_hline(y=0, line_dash="solid", line_color="black")
    
    fig.update_layout(
        title='Backtest Trades',
        xaxis_title='Exit Date',
        yaxis_title='P&L ($)',
        height=500,
        showlegend=False
    )
    
    return fig

def plot_backtest_equity_curve(daily_performance):
    """Create an interactive equity curve plot from backtest daily performance"""
    if not daily_performance:
        return None
    
    # Convert to DataFrame
    df = pd.DataFrame(daily_performance)
    
    # Convert date strings to datetime
    df['date'] = pd.to_datetime(df['date'])
    
    fig = go.Figure()
    
    # Add equity curve
    fig.add_trace(go.Scatter(
        x=df['date'],
        y=df['equity'],
        mode='lines',
        name='Equity',
        line=dict(color='blue', width=2)
    ))
    
    # Add drawdown
    fig.add_trace(go.Scatter(
        x=df['date'],
        y=df['drawdown'],
        mode='lines',
        name='Drawdown %',
        line=dict(color='red', width=1, dash='dot'),
        yaxis='y2'
    ))
    
    # Update layout with two y-axes
    fig.update_layout(
        title='Backtest Equity Curve and Drawdown',
        xaxis_title='Date',
        yaxis_title='Equity ($)',
        yaxis2=dict(
            title='Drawdown (%)',
            overlaying='y',
            side='right',
            range=[min(df['drawdown']) * 1.5, 1],  # Adjust range for better visualization
            showgrid=False
        ),
        height=500,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    return fig

def main():
    """Main dashboard application"""
    # Sidebar with authentication
    if not check_password():
        st.stop()  # Do not proceed if password is incorrect
    
    # Set page config
    st.set_page_config(
        page_title="Trading Strategy Dashboard",
        page_icon="📈",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Dashboard title
    st.title("📊 Advanced Trading Strategy Dashboard")
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Select Page",
        ["📈 Live Performance", "🔄 Backtest Analysis", "📊 Market Analysis", "📝 System Logs"]
    )
    
    # Refresh button
    if st.sidebar.button("🔄 Refresh Data"):
        st.cache_data.clear()
        st.experimental_rerun()
    
    # Last updated timestamp
    st.sidebar.write(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Load data
    live_data = load_trading_data()
    backtest_data = load_backtest_data()
    
    # =====================
    # Live Performance Page
    # =====================
    if page == "📈 Live Performance":
        st.header("Live Trading Performance")
        
        # Compute metrics
        live_data_processed, live_metrics = compute_strategy_metrics(live_data)
        
        # Create columns for metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total P&L", f"${live_metrics['Total PnL']:.2f}", f"{live_metrics['Return (%)']:.2f}%")
        
        with col2:
            st.metric("Win Rate", f"{live_metrics['Win Rate (%)']:.2f}%")
        
        with col3:
            st.metric("Sharpe Ratio", f"{live_metrics['Sharpe Ratio']:.2f}")
        
        with col4:
            st.metric("Max Drawdown", f"{live_metrics['Max Drawdown (%)']:.2f}%")
        
        # Equity curve
        st.subheader("Equity Curve")
        equity_fig = plot_equity_curve(live_data_processed)
        st.plotly_chart(equity_fig, use_container_width=True)
        
        # PnL distribution and Win Rate over time in two columns
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("P&L Distribution")
            pnl_fig = plot_pnl_distribution(live_data_processed)
            st.plotly_chart(pnl_fig, use_container_width=True)
        
        with col2:
            st.subheader("Win Rate Over Time")
            win_rate_fig = plot_win_rate_over_time(live_data_processed)
            if win_rate_fig:
                st.plotly_chart(win_rate_fig, use_container_width=True)
            else:
                st.info("Win rate data not available")
        
        # Trading activity
        st.subheader("Trading Activity")
        activity_fig = plot_trading_activity(live_data_processed)
        if activity_fig:
            st.plotly_chart(activity_fig, use_container_width=True)
        else:
            st.info("Trading activity data not available")
        
        # AI-generated performance insights
        st.subheader("Strategy Insights")
        with st.spinner("Generating strategy insights..."):
            market_data = load_market_data(['^GSPC'])  # Load market data for comparison
            insights = generate_ai_insights(live_metrics, backtest_data.get("metrics", {}), market_data)
            st.markdown(insights)
    
    # =====================
    # Backtest Analysis Page
    # =====================
    elif page == "🔄 Backtest Analysis":
        st.header("Backtest Analysis")
        
        # Strategy metadata
        strategy_name = backtest_data.get("metadata", {}).get("strategy", {}).get("name", "Unknown Strategy")
        parameters = backtest_data.get("metadata", {}).get("strategy", {}).get("parameters", {})
        symbols = backtest_data.get("metadata", {}).get("data", {}).get("symbols", [])
        start_date = backtest_data.get("metadata", {}).get("data", {}).get("start_date", "")
        end_date = backtest_data.get("metadata", {}).get("data", {}).get("end_date", "")
        
        # Strategy info
        st.subheader("Strategy Information")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.write(f"**Strategy:** {strategy_name}")
            st.write(f"**Period:** {start_date} to {end_date}")
        
        with col2:
            st.write(f"**Symbols:** {', '.join(symbols)}")
        
        with col3:
            st.write(f"**Trades:** {backtest_data.get('metrics', {}).get('total_trades', 0)}")
        
        # Key metrics
        metrics = backtest_data.get("metrics", {})
        benchmark = backtest_data.get("benchmark", {})
        
        st.subheader("Key Performance Metrics")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Final Value", f"${metrics.get('final_value', 0):.2f}", 
                    f"{metrics.get('pnl_percent', 0):.2f}%")
        
        with col2:
            st.metric("Benchmark", f"{benchmark.get('symbol', '')}",
                    f"{benchmark.get('return_percent', 0):.2f}%")
        
        with col3:
            alpha = metrics.get('pnl_percent', 0) - benchmark.get('return_percent', 0)
            st.metric("Alpha", f"{alpha:.2f}%")
        
        with col4:
            st.metric("Sharpe Ratio", f"{metrics.get('sharpe', 0):.2f}")
        
        # Additional metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Max Drawdown", f"{metrics.get('max_drawdown', 0):.2f}%")
        
        with col2:
            st.metric("Win Rate", f"{metrics.get('win_rate', 0):.2f}%")
        
        with col3:
            st.metric("Avg. Trade", f"${metrics.get('avg_trade_pnl', 0):.2f}")
        
        with col4:
            st.metric("System Quality", f"{metrics.get('sqn', 0):.2f}")
        
        # Equity curve from backtest
        st.subheader("Equity Curve")
        equity_fig = plot_backtest_equity_curve(backtest_data.get("daily_performance", []))
        if equity_fig:
            st.plotly_chart(equity_fig, use_container_width=True)
        else:
            st.info("Daily performance data not available")
        
        # Trade scatter plot
        st.subheader("Individual Trades")
        trades_fig = plot_backtest_trades(backtest_data.get("trades", []))
        if trades_fig:
            st.plotly_chart(trades_fig, use_container_width=True)
        else:
            st.info("Trade data not available")
        
        # Strategy parameters
        st.subheader("Strategy Parameters")
        
        # Create a DataFrame from parameters for better display
        if parameters:
            param_df = pd.DataFrame(list(parameters.items()), columns=["Parameter", "Value"])
            st.dataframe(param_df, use_container_width=True)
        else:
            st.info("No parameters available")
    
    # =====================
    # Market Analysis Page
    # =====================
    elif page == "📊 Market Analysis":
        st.header("Market Analysis")
        
        # Get symbols from backtest data, or use defaults
        default_symbols = backtest_data.get("metadata", {}).get("data", {}).get("symbols", [])
        
        if not default_symbols:
            default_symbols = ['SPY', 'QQQ', 'AAPL', 'MSFT', 'AMZN']
        
        # Symbol selection
        selected_symbols = st.multiselect(
            "Select Symbols to Analyze",
            options=default_symbols + ['SPY', 'QQQ', 'IWM', 'DIA', 'AAPL', 'MSFT', 'AMZN', 'GOOGL', 'META', 'TSLA'],
            default=default_symbols[:3]  # Default to first 3 symbols
        )
        
        # Time period selection
        period_options = {
            '1 Month': '1mo',
            '3 Months': '3mo',
            '6 Months': '6mo',
            '1 Year': '1y',
            '2 Years': '2y',
            '5 Years': '5y'
        }
        
        selected_period = st.select_slider(
            "Select Time Period",
            options=list(period_options.keys()),
            value='6 Months'
        )
        
        # Load market data
        if selected_symbols:
            with st.spinner("Loading market data..."):
                market_data = load_market_data(selected_symbols, period=period_options[selected_period])
            
            # Create price chart
            st.subheader("Price Comparison")
            
            # Normalize prices to starting point
            normalized_data = {}
            for symbol, data in market_data.items():
                if not data.empty:
                    normalized_data[symbol] = data['Close'] / data['Close'].iloc[0] * 100
            
            if normalized_data:
                # Convert to DataFrame for plotting
                normalized_df = pd.DataFrame(normalized_data)
                
                # Create plot
                fig = go.Figure()
                
                for symbol in normalized_df.columns:
                    fig.add_trace(go.Scatter(
                        x=normalized_df.index,
                        y=normalized_df[symbol],
                        mode='lines',
                        name=symbol
                    ))
                
                fig.update_layout(
                    title=f"Normalized Price Performance (Base=100)",
                    xaxis_title="Date",
                    yaxis_title="Normalized Price",
                    height=500,
                    yaxis=dict(
                        tickformat=".2f"
                    )
                )
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No market data available")
            
            # Correlation matrix
            st.subheader("Correlation Matrix")
            
            # Create correlation matrix
            returns_data = {}
            for symbol, data in market_data.items():
                if not data.empty and len(data) > 1:
                    returns_data[symbol] = data['Close'].pct_change().dropna()
            
            if len(returns_data) > 1:
                # Convert to DataFrame
                returns_df = pd.DataFrame(returns_data)
                
                # Calculate correlation matrix
                corr_matrix = returns_df.corr()
                
                # Create heatmap
                fig = go.Figure(data=go.Heatmap(
                    z=corr_matrix.values,
                    x=corr_matrix.columns,
                    y=corr_matrix.index,
                    colorscale='RdBu',
                    zmin=-1,
                    zmax=1,
                    colorbar=dict(title="Correlation")
                ))
                
                fig.update_layout(
                    title="Correlation Matrix of Daily Returns",
                    height=500
                )
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Insufficient data for correlation analysis")
            
            # Volatility comparison
            st.subheader("Volatility Comparison")
            
            # Calculate volatility (20-day rolling standard deviation of returns)
            volatility_data = {}
            for symbol, data in market_data.items():
                if not data.empty and len(data) > 20:
                    volatility_data[symbol] = data['Close'].pct_change().rolling(window=20).std() * np.sqrt(252) * 100  # Annualized volatility (%)
            
            if volatility_data:
                # Convert to DataFrame
                volatility_df = pd.DataFrame(volatility_data)
                
                # Create plot
                fig = go.Figure()
                
                for symbol in volatility_df.columns:
                    fig.add_trace(go.Scatter(
                        x=volatility_df.index,
                        y=volatility_df[symbol],
                        mode='lines',
                        name=symbol
                    ))
                
                fig.update_layout(
                    title="20-Day Rolling Volatility (Annualized %)",
                    xaxis_title="Date",
                    yaxis_title="Volatility (%)",
                    height=500
                )
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Insufficient data for volatility analysis")
            
            # Volume analysis
            st.subheader("Volume Analysis")
            
            volume_data = {}
            for symbol, data in market_data.items():
                if not data.empty and 'Volume' in data.columns:
                    # Normalize volume by dividing by its mean
                    volume_data[symbol] = data['Volume'] / data['Volume'].mean()
            
            if volume_data:
                # Convert to DataFrame
                volume_df = pd.DataFrame(volume_data)
                
                # Create plot
                fig = go.Figure()
                
                for symbol in volume_df.columns:
                    fig.add_trace(go.Scatter(
                        x=volume_df.index,
                        y=volume_df[symbol],
                        mode='lines',
                        name=symbol
                    ))
                
                fig.update_layout(
                    title="Relative Volume (1.0 = Average)",
                    xaxis_title="Date",
                    yaxis_title="Relative Volume",
                    height=500
                )
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Volume data not available")
        else:
            st.info("Please select at least one symbol to analyze")
    
    # =====================
    # System Logs Page
    # =====================
    elif page == "📝 System Logs":
        st.header("System Logs")
        
        # Load log files
        log_lines = load_log_file()
        
        if log_lines:
            # Parse log entries
            log_df = parse_log_entries(log_lines)
            
            # Log filtering
            st.subheader("Log Filtering")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Filter by log level
                if 'level' in log_df.columns:
                    levels = log_df['level'].unique().tolist()
                    selected_levels = st.multiselect(
                        "Filter by Log Level",
                        options=levels,
                        default=levels
                    )
                    
                    if selected_levels:
                        log_df = log_df[log_df['level'].isin(selected_levels)]
            
            with col2:
                # Filter by text search
                if 'message' in log_df.columns:
                    search_term = st.text_input("Search Logs", "")
                    
                    if search_term:
                        log_df = log_df[log_df['message'].str.contains(search_term, case=False)]
            
            # Display logs
            st.subheader("Log Entries")
            
            if not log_df.empty:
                # Sort by timestamp descending
                log_df = log_df.sort_values('timestamp', ascending=False)
                
                # Custom formatter for logs
                def format_log(row):
                    level_color = {
                        'INFO': 'blue',
                        'WARNING': 'orange',
                        'ERROR': 'red',
                        'CRITICAL': 'purple',
                        'DEBUG': 'green'
                    }
                    
                    color = level_color.get(row['level'], 'black')
                    
                    return f"""
                    <div style="margin-bottom: 8px; padding: 8px; border-left: 3px solid {color}; background-color: rgba(0,0,0,0.05);">
                        <div style="color: gray; font-size: 0.8em;">{row['timestamp']}</div>
                        <div style="color: {color}; font-weight: bold;">{row['level']}</div>
                        <div style="margin-top: 4px;">{row['message']}</div>
                    </div>
                    """
                
                # Display logs
                html_logs = ''.join(log_df.apply(format_log, axis=1).tolist())
                st.markdown(f'<div style="max-height: 600px; overflow-y: auto;">{html_logs}</div>', unsafe_allow_html=True)
            else:
                st.info("No log entries match the current filters")
            
            # Log summary
            if 'level' in log_df.columns:
                st.subheader("Log Summary")
                
                log_counts = log_df['level'].value_counts().reset_index()
                log_counts.columns = ['Level', 'Count']
                
                # Create bar chart
                fig = px.bar(
                    log_counts, 
                    x='Level', 
                    y='Count',
                    color='Level',
                    color_discrete_map={
                        'INFO': 'blue',
                        'WARNING': 'orange',
                        'ERROR': 'red',
                        'CRITICAL': 'purple',
                        'DEBUG': 'green'
                    }
                )
                
                fig.update_layout(
                    title="Log Level Distribution",
                    xaxis_title="Log Level",
                    yaxis_title="Count",
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No log entries available")
    
    # Footer
    st.markdown("---")
    st.markdown("Trading Strategy Dashboard v2.0 | © 2025")

if __name__ == "__main__":
    main()