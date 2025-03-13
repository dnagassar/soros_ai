import backtrader as bt

class SentimentStrategy(bt.Strategy):
    params = dict(
        stop_loss=0.03,   # 3% stop loss
        take_profit=0.06  # 6% take profit
    )

    def __init__(self):
        self.dataclose = self.datas[0].close
        self.sma = bt.indicators.SimpleMovingAverage(self.datas[0], period=15)
        self.order = None

    def next(self):
        if self.order:
            return

        # Example: buy when price is above SMA; sell when below
        if self.dataclose[0] > self.sma[0]:
            self.order = self.buy_bracket(
                size=100,
                stopprice=self.dataclose[0] * (1 - self.p.stop_loss),
                limitprice=self.dataclose[0] * (1 + self.p.take_profit)
            )
        elif self.dataclose[0] < self.sma[0]:
            self.order = self.sell_bracket(
                size=100,
                stopprice=self.dataclose[0] * (1 + self.p.stop_loss),
                limitprice=self.dataclose[0] * (1 - self.p.take_profit)
            )

if __name__ == "__main__":
    cerebro = bt.Cerebro()
    cerebro.addstrategy(SentimentStrategy)
    data = bt.feeds.YahooFinanceCSVData(dataname='../data/historical_prices.csv')
    cerebro.adddata(data)
    cerebro.run()
    cerebro.plot()
