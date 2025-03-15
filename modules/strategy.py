# modules/strategy.py
import backtrader as bt

class AdaptiveSentimentStrategy(bt.Strategy):
    params = dict(
        signal=0,          # Aggregated signal (pass as a parameter)
        stop_loss=0.03,    # 3% stop loss
        take_profit=0.06,  # 6% take profit
        atr_period=14,     # Period for ATR calculation
        risk_factor=0.01,  # Risk 1% of capital per trade
    )

    def __init__(self):
        self.dataclose = self.datas[0].close
        self.atr = bt.indicators.ATR(self.datas[0], period=self.p.atr_period)
        self.order = None

    def next(self):
        # If there is an open order, do nothing
        if self.order:
            return
        
        atr = self.atr[0]
        cash = self.broker.get_cash()
        risk_amount = cash * self.p.risk_factor
        position_size = risk_amount / atr
        position_size = int(position_size)

        # Use the aggregated signal (passed as a parameter) to decide trade direction
        if self.p.signal > 0:
            self.order = self.buy_bracket(
                size=position_size,
                stopprice=self.dataclose[0] - (atr * 2),
                limitprice=self.dataclose[0] + atr * 2
            )
        elif self.p.signal < 0:
            self.order = self.sell_bracket(
                size=position_size,
                stopprice=self.dataclose[0] + atr,
                limitprice=self.dataclose[0] - atr
            )

if __name__ == "__main__":
    cerebro = bt.Cerebro()
    cerebro.addstrategy(AdaptiveSentimentStrategy, signal=1)  # Dummy signal for testing
    data = bt.feeds.YahooFinanceCSVData(dataname='../data/historical_prices.csv')
    cerebro.adddata(data)
    cerebro.run()
    cerebro.plot()
