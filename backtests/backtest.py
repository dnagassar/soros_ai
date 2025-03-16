import backtrader as bt
from modules.asset_selector import select_top_assets
from modules.data_acquisition import fetch_price_data
from modules.strategy import AdaptiveSentimentStrategy
import datetime
import matplotlib.pyplot as plt
import yfinance as yf

def fetch_benchmark_data(symbol, start, end, initial_cash):
    benchmark = yf.download(symbol, start=start, end=end)['Close']
    return benchmark / benchmark.iloc[0] * initial_cash

def run_backtest_with_benchmark(start_date, end_date, initial_cash=100000, assets_to_trade=None, benchmark_symbol='^GSPC'):
    cerebro = bt.Cerebro()
    cerebro.addstrategy(AdaptiveSentimentStrategy)
    cerebro.broker.setcash(initial_cash)

    if assets_to_trade is None:
        assets_to_trade = select_top_assets(n=5)

    for symbol in assets_to_trade:
        data = fetch_price_data(symbol, start_date, end_date)
        data_file = f'data/{symbol}_prices.csv'
        data.to_csv(data_file)

        data_feed = bt.feeds.YahooFinanceCSVData(
            dataname=data_file,
            fromdate=datetime.datetime.strptime(start_date, '%Y-%m-%d'),
            todate=datetime.datetime.strptime(end_date, '%Y-%m-%d'),
            reverse=False
        )
        cerebro.adddata(data_feed, name=symbol)

    results = cerebro.run()
    portfolio_value = cerebro.broker.getvalue()
    pnl = portfolio_value - initial_cash

    print(f'Final Portfolio Value: ${portfolio_value:.2f}')
    print(f'PnL: ${pnl:.2f}')

    # Fetch benchmark data
    benchmark = fetch_benchmark_data(benchmark_symbol, start_date, end_date, initial_cash)

    # Plot strategy vs benchmark
    plt.figure(figsize=(10, 6))
    plt.plot(benchmark.index, benchmark, label='S&P 500', linewidth=2)
    plt.axhline(portfolio_value, color='green', linestyle='--', label='Strategy Final Value', linewidth=2)
    plt.title('Your Strategy vs S&P 500 Benchmark')
    plt.xlabel('Date')
    plt.ylabel('Portfolio Value ($)')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Backtrader visualization (this might open a separate window)
    cerebro.plot()

if __name__ == '__main__':
    run_backtest_with_benchmark('2022-01-01', '2023-01-01')
