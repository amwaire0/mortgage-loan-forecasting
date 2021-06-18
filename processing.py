import pandas as pd
from scipy.stats import spearmanr

class DataProcessing:
    def __init__(self):
        self.volume, self.rate, self.index, self.prices, self.usd, self.euro, self.oil, self.yandex1, self.yandex2 = self.read_data()
        self.lagged_df = self.correlation_analysis()
        
    def read_data(self):
        df_volume = pd.read_csv('data/cb.csv', index_col='Date')
        df_rate = pd.read_csv('data/key_rate.csv', index_col='Date')
        df_index = pd.read_csv('data/индекс_потр_цен.csv', index_col='Date')
        df_prices = pd.read_csv('data/средние_цены_кв.csv', index_col='Date')
        df_usd = pd.read_csv('data/usd_rub.csv', index_col='Date')
        df_euro = pd.read_csv('data/euro_rub.csv', index_col='Date')
        df_oil = pd.read_csv('data/crude-oil.csv', index_col='Date')
        df_yandex1 = pd.read_csv('data/yand_month.csv', index_col='date')
        df_yandex2 = pd.read_csv('data/yand_mortg_month.csv', index_col='date')
        volume = df_volume['Money'].values
        rate = df_rate['key_rate'].values
        index = df_index['CPI'].values
        prices = df_prices['flats_pric'].values
        usd = df_usd['usd'].values
        euro = df_euro['euro'].values
        oil = df_oil['crude_oil'].values
        yandex1 = df_yandex1['query'].values
        yandex2 = df_yandex2['query'].values
        return (volume, rate, index, prices, usd, euro, oil, yandex1, yandex2)
    
    def correlation_analysis(self):
        lagged = []
        for predictor in [self.rate, self.index, self.prices, self.usd, self.euro, self.oil, self.yandex1, self.yandex2]:
            max_cor, ind_max = 0, 0
            for i in range(4):
                if i != 0:
                    coef, p = spearmanr(predictor[3-i:-i], self.volume)
                    if abs(coef) > abs(max_cor):
                        max_cor = coef
                        ind_max = i
                else:
                    coef, p = spearmanr(predictor[3:], self.volume)
            lagged.append(predictor[3-ind_max:-ind_max])
        d = {'Volume of mortgage loans': self.volume, 'Russian Central Bank Key rate': lagged[0], 'Consumer price index': lagged[1], 
             'Average apartment price': lagged[2], 'Dollar/ruble exchange rate': lagged[3], 'Euro/ruble exchange rate': lagged[4], 
             'Crude oil price': lagged[5], 'Yandex Query 1': lagged[6], 'Yandex Query 2': lagged[7]}
        lagged_df = pd.DataFrame(d)
        df_volume = pd.read_csv('data/cb.csv', index_col='Date')
        lagged_df['date'] = df_volume.index
        lagged_df = lagged_df.set_index('date')
        return lagged_df