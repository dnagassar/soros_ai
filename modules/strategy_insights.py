# modules/strategy_insights.py
import openai
from config import OPENAI_API_KEY

# Set your OpenAI API key
openai.api_key = OPENAI_API_KEY

def get_strategy_insights(performance_data):
    """
    Given a dictionary of performance data, generates actionable trading strategy insights using an LLM.
    
    performance_data should include keys such as:
        - 'PnL': Profit and Loss information.
        - 'signal_accuracy': Accuracy percentage of trading signals.
        - 'drawdown': Maximum drawdown observed.
        - 'number_of_trades': Total number of trades executed.
        - 'avg_position_size': Average position size used.
    
    Returns:
        A string containing the LLM-generated recommendations.
    """
    prompt = (
        f"Our trading system over the past 24 hours reported the following metrics:\n"
        f"- PnL: {performance_data.get('PnL', 'N/A')}\n"
        f"- Signal Accuracy: {performance_data.get('signal_accuracy', 'N/A')}\n"
        f"- Drawdown: {performance_data.get('drawdown', 'N/A')}\n"
        f"- Number of Trades: {performance_data.get('number_of_trades', 'N/A')}\n"
        f"- Average Position Size: {performance_data.get('avg_position_size', 'N/A')}\n\n"
        "Based on these metrics, please provide detailed, actionable suggestions for improving risk management, "
        "signal generation, position sizing, and overall strategy performance. Focus on adaptive methods and any adjustments "
        "that could help the system consistently outperform market benchmarks."
    )
    
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",  # or another model you have access to
        messages=[
            {"role": "system", "content": "You are a trading strategy consultant."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7,
        max_tokens=200,
    )
    
    insight = response["choices"][0]["message"]["content"].strip()
    return insight

if __name__ == "__main__":
    # Sample performance data for testing purposes.
    sample_data = {
        "PnL": "$500",
        "signal_accuracy": "75%",
        "drawdown": "2%",
        "number_of_trades": "25",
        "avg_position_size": "100 shares"
    }
    insights = get_strategy_insights(sample_data)
    print("Strategy Insights:")
    print(insights)
