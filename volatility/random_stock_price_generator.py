#!/usr/bin/env python

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class RandomStockPrices:
    def __init__(self, start_date='1956-01-01', years=1, num_assets=1000, constant_dist=False):

        self.start_date = start_date
        self.num_assets = num_assets
        self.years = years
        if constant_dist:
            self._create_prices_constant_dist()
        else:
            self._create_prices()

    def _create_prices_constant_dist(self):
        dates = pd.bdate_range(start=self.start_date, periods=self.years*252)
        days = len(dates)
        prices = np.zeros((days, self.num_assets))
        returns = np.zeros((days, self.num_assets))
        initial_prices = 5.0 + np.random.gamma(2, 30, self.num_assets)
        distributions = list(zip(
            np.random.normal(0.10, 0.10, self.num_assets),
            0.04 + np.random.gamma(6, 2, self.num_assets) / 100))
        self.distributions = distributions
        for i in range(self.num_assets):
            mean, std = distributions[i]
            dt = 1.0/252
            sr_dt = np.sqrt(dt)
            drift = (mean - (std**2)/2)*dt
            diffusion = std * sr_dt * np.random.randn(days)
            returns[:,i] = drift + diffusion
            exp_returns = np.exp(returns[:,i])
            prices[:,i] = initial_prices[i] * np.cumprod(exp_returns)

        tickers = ['TICK{}'.format(i) for i in range(self.num_assets)]
        self.prices = pd.DataFrame(prices, index=dates, columns=tickers)
        self.returns = pd.DataFrame(returns, index=dates, columns=tickers)

    def _create_prices(self):
        dates = pd.bdate_range(start=self.start_date, periods=self.years*252)
        days = len(dates)
        prices = np.zeros((self.years*252, self.num_assets))
        returns = np.zeros((self.years*252, self.num_assets))
        initial_prices = 5.0 + np.random.gamma(2, 30, self.num_assets)
        yearly_distributions = np.random.normal(0.10, 0.20, self.years)

        for i in range(self.years):
            yr_mean = yearly_distributions[i]
            distributions = list(zip(
                np.random.normal(yr_mean, 0.05, self.num_assets),
                0.04 + np.random.gamma(6, 2, self.num_assets) / 100))
            for j in range(self.num_assets):
                mean, std = distributions[j]
                dt = 1.0/252.0
                sr_dt = np.sqrt(dt)
                drift = (mean - (std**2)/2) * dt
                diffusion = np.random.randn(252) * sr_dt * std
                start_idx = i*252
                end_idx = (i+1)*252
                returns[start_idx:end_idx,j] = drift + diffusion

        for j in range(self.num_assets):
            price = initial_prices[j]
            exp_returns = np.exp(returns[:,j])
            prices[:,j] = price * np.cumprod(exp_returns)

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

