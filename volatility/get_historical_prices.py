import os
import pandas as pd
from tqdm import tqdm

class HistoricalPrices:
    def __init__(self, start_date=None, num_assets=967, num_days=None, prices_dir='prices'):
        self.num_days = num_days
        self.num_assets = num_assets
        self.start_date = start_date
        self.price_dir = prices_dir

    def get_prices(self):
        prices = pd.DataFrame()
        for filename in tqdm(os.listdir(self.price_dir)):
            filename = str('{}/{}'.format(self.price_dir, filename))
            prices = prices.append(pd.read_csv(filename))

        prices.columns = ['ticker', 'date', 'high', 'low', 'adjclose', 'volume']
        self.prices = prices.set_index('date')



if __name__ == '__main__':
    hp = HistoricalPrices()
    hp.get_prices()
    print(hp.prices.head())


