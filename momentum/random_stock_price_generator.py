#!/usr/bin/env python

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class RandomStockPrices:
    def __init__(self, start_date='2018-01-01', num_days=252, num_assets=1000,
                 mean_mean=0.05, mean_std=0.05):

        self.start_date = start_date
        self.num_days = num_days
        self.num_assets = num_assets
        self.mean_mean=mean_mean
        self.mean_std = mean_std
        self._create_prices()

    def _create_prices(self):
        dates = pd.bdate_range(start=self.start_date, periods=self.num_days)
        days = len(dates)
        prices = np.zeros((days, self.num_assets))
        returns = np.zeros((days, self.num_assets))
        initial_prices = 5.0 + np.random.gamma(2, 30, self.num_assets)
        #plt.hist(initial_prices)
        #plt.hist(initial_prices)
        #plt.show()
        distributions = list(zip(
            np.random.normal(self.mean_mean, self.mean_std, self.num_assets),
            0.04 + np.random.gamma(6, 2, self.num_assets) / 100))
        self.distributions = distributions
        for i in range(self.num_assets):
            mean, std = distributions[i]
            mean /= 252
            std /= np.sqrt(252)
            returns[:,i] = np.random.normal(mean, std, days)
            price = initial_prices[i]
            for j in range(days):
                price = price * (1+returns[j,i])
                prices[j,i] = price

        tickers = ['TICK{}'.format(i) for i in range(self.num_assets)]
        self.prices = pd.DataFrame(prices, index=dates, columns=tickers)
        self.returns = pd.DataFrame(returns, index=dates, columns=tickers)

def tester():
    mc = RandomStockPrices()
    ax = mc.prices.head(100).plot(ylim=(2,40))
    ax.tick_params(labelright=True, right=True)
    plt.show()



if __name__ == '__main__':
    #np.random.seed(0)
    tester()
    exit()
    sd1 = 0.18 / np.sqrt(252)
    sd2 = 0.12 / np.sqrt(252)
    mu1 = 0.12 / 252
    mu2 = 0.08 / 252
    rs1 = np.random.randn(100)
    rs2 = np.random.randn(100)
    corr = 0.6
    stock1_return = mu1 + rs1*sd1
    stock2_return = mu2 + rs2*sd2
    stock2_return = stock2_return + corr*stock1_return
    stock1_prices = []
    stock2_prices = []
    for i in range(100):
        if i == 0:
            stock1_prices.append(100*(1+stock1_return[i]))
            stock2_prices.append(100*(1+stock2_return[i]))
            continue

        stock1_prices.append(stock1_prices[i-1] * (1+stock1_return[i]))
        stock2_prices.append(stock2_prices[i-1]*(1+stock2_return[i]))

    realized_corr = np.corrcoef(stock1_prices, stock2_prices)
    print(realized_corr)
    dates = pd.bdate_range(start='2018-01-01', periods=100)
    df = pd.DataFrame(data={'stock1_rtn': stock1_return, 'stock2_rtn':stock2_return, 'stock1_price': stock1_prices, 'stock2_price': stock2_prices}, index=dates)

    df[['stock1_price', 'stock2_price']].plot()
    plt.show()

