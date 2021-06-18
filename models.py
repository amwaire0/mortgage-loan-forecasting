from processing import DataProcessing

import numpy as np
import statsmodels.api as sm
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense

def MAPE(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

class Baseline_predictor(DataProcessing):
    def __init__(self):
        DataProcessing.__init__(self)
        self.train, self.test = self.train_test_split()
        self.mape = self.model()
        
    def train_test_split(self):
        df = self.lagged_df
        size = int(len(df) * 0.7)
        df_train, df_test = df[:size], df[size:]
        return (df_train, df_test)
    
    def model(self):
        history = [x for x in self.train['Volume of mortgage loans'].values]
        predictions = list()
        for t in range(len(self.test)):
            output = history[-1]
            predictions.append(output)
            obs = self.test['Volume of mortgage loans'].values[t]
            history.append(obs)
        mape = MAPE(self.test['Volume of mortgage loans'], predictions)
        return (mape)

class ARIMA_predictor(DataProcessing):
    def __init__(self, p, d, q):
        DataProcessing.__init__(self)
        self.train, self.test = self.train_test_split()
        self.p = p
        self.q = q
        self.d = d
        self.mape = self.model()
        
    def train_test_split(self):
        df = self.lagged_df
        size = int(len(df) * 0.7)
        df_train, df_test = df[:size], df[size:]
        return (df_train, df_test)
    
    def model(self):
        history = [x for x in self.train['Volume of mortgage loans'].values]
        predictions = list()
        for t in range(len(self.test)):
            model = sm.tsa.statespace.SARIMAX(history, order=(self.p,self.d,self.q), seasonal_order=(0,0,0,0))
            model_fit = model.fit()
            output = model_fit.predict(start=len(self.train)+t, end=len(self.train)+1+t)
            yhat = output[0]
            predictions.append(yhat)
            obs = self.test['Volume of mortgage loans'].values[t]
            history.append(obs)
        mape = MAPE(self.test['Volume of mortgage loans'], predictions)
        return (mape)
    
class ARIMAX_predictor(DataProcessing):
    def __init__(self, p, d, q):
        DataProcessing.__init__(self)
        self.train, self.test = self.train_test_split()
        self.p = p
        self.q = q
        self.d = d
        self.mape = self.model()
        
    def train_test_split(self):
        df = self.lagged_df
        size = int(len(df) * 0.7)
        df_train, df_test = df[:size], df[size:]
        return (df_train, df_test)
    
    def model(self):
        history = [x for x in self.train['Volume of mortgage loans'].values]
        ex1 = [x for x in self.train['Russian Central Bank Key rate'].values]
        ex2 = [x for x in self.train['Dollar/ruble exchange rate'].values]
        ex3 = [x for x in self.train['Yandex Query 1'].values]
        ex = np.transpose(np.array([ex1, ex2, ex3]))
        predictions = list()
        for t in range(len(self.test)):
            model = sm.tsa.statespace.SARIMAX(history, exog=ex, order=(self.p,self.d,self.q), seasonal_order=(0,0,0,0))
            model_fit = model.fit()
            exog1, exog2, exog3 = [], [], []
            exog1.append(self.test['Russian Central Bank Key rate'].values[t])
            exog2.append(self.test['Dollar/ruble exchange rate'].values[t])
            exog3.append(self.test['Yandex Query 1'].values[t])
            exog = np.transpose(np.array([exog1, exog2, exog3]))
            output = model_fit.predict(start=len(self.train)+t, end=len(self.train)+t, exog=exog)
            predictions.append(output[0])
            obs = self.test['Volume of mortgage loans'].values[t]
            history.append(obs)
            ex = np.vstack((ex, exog))
        mape = MAPE(self.test['Volume of mortgage loans'], predictions)
        return (mape)
    
class LSTM_predictor(DataProcessing):
    def __init__(self, NFILTERS, BATCH_SIZE, NB_EPOCHS, w_len, val_split):
        DataProcessing.__init__(self)
        self.w_len = w_len
        self.train, self.test, self.scaler = self.train_test_split()
        self.NFILTERS = NFILTERS
        self.NB_EPOCHS = NB_EPOCHS
        self.BATCH_SIZE = BATCH_SIZE
        self.val_split = val_split
        self.mape = self.model()
        
    def train_test_split(self):
        df = self.lagged_df
        df = df[['Volume of mortgage loans']]
        size = int(len(df) * 0.7)
        df_val = df.values
        df_val = df_val.astype('float32')
        scaler = MinMaxScaler(feature_range=(0, 1))
        df_val = scaler.fit_transform(df_val)
        df_train, df_test = df_val[0:size, :], df_val[(size-self.w_len):len(df_val), :]
        np.random.seed(7)
        return (df_train, df_test, scaler)
    
    def create_dataset(self, dataset, look_back=1):
        dataX, dataY = [], []
        for i in range(len(dataset)-look_back):
            a = dataset[i:(i+look_back), 0]
            dataX.append(a)
            b = dataset[i+look_back, 0]
            dataY.append(b)
        return np.array(dataX), np.array(dataY)
    
    def model(self):
        trainX, trainY = self.create_dataset(self.train, look_back=self.w_len)
        testX, testY = self.create_dataset(self.test, look_back=self.w_len)
        trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
        testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
        model = Sequential()
        model.add(LSTM(self.NFILTERS, input_shape=(1, self.w_len), return_sequences=True))
        model.add(LSTM(self.NFILTERS))
        model.add(Dense(1))
        model.compile(loss='mae', optimizer='adam')
        model.fit(trainX, trainY, epochs = self.NB_EPOCHS, batch_size=self.BATCH_SIZE, verbose=2, validation_split=self.val_split)
        predictions = list()
        for t in range(len(testX)):
            X = testX[t]
            yhat = model.predict(np.reshape(X, (1,1,self.w_len)))
            predictions.append(yhat[0])
        testPredict = self.scaler.inverse_transform([np.concatenate(predictions)])
        testY = np.hstack(testY)
        testY = self.scaler.inverse_transform([np.concatenate([np.expand_dims(i,axis=0) for i in testY])])
        mape = MAPE(testY[0], testPredict[0])
        return (mape)
    
class Transfer_learning_predictor(DataProcessing):
    def __init__(self, NFILTERS, BATCH_SIZE, NB_EPOCHS, w_len, val_split):
        DataProcessing.__init__(self)
        self.w_len = w_len
        self.train, self.test, self.scaler = self.train_test_split()
        self.NFILTERS = NFILTERS
        self.NB_EPOCHS = NB_EPOCHS
        self.BATCH_SIZE = BATCH_SIZE
        self.val_split = val_split
        self.model_aux = self.aux_model()
        self.mape = self.model()
        
    def create_dataset(self, dataset, look_back=1):
        dataX, dataY = [], []
        for i in range(len(dataset)-look_back):
            a = dataset[i:(i+look_back), 0]
            dataX.append(a)
            b = dataset[i+look_back, 0]
            dataY.append(b)
        return np.array(dataX), np.array(dataY)
        
    def aux_model(self):
        aux = pd.read_csv('data/EURRUB.csv', index_col='Date')
        aux = aux[['Close']]
        df_val_aux = aux.values
        df_val_aux = df_val_aux.astype('float32')
        scaler_aux = MinMaxScaler(feature_range=(0, 1))
        df_val_aux = scaler_aux.fit_transform(df_val_aux)
        trainX_aux, trainY_aux = self.create_dataset(df_val_aux, look_back=self.w_len)
        trainX_aux = np.reshape(trainX_aux, (trainX_aux.shape[0], 1, trainX_aux.shape[1]))
        model_aux = Sequential()
        model_aux.add(LSTM(self.NFILTERS, input_shape=(1, self.w_len), return_sequences=True))
        model_aux.add(LSTM(self.NFILTERS))
        model_aux.add(Dense(1))
        model_aux.compile(loss='mae', optimizer='adam')
        model_aux.fit(trainX_aux, trainY_aux, epochs = self.NB_EPOCHS, batch_size=self.BATCH_SIZE, verbose=2, validation_split=self.val_split)
        return model_aux
        
    def train_test_split(self):
        df = self.lagged_df
        df = df[['Volume of mortgage loans']]
        size = int(len(df) * 0.7)
        df_val = df.values
        df_val = df_val.astype('float32')
        scaler = MinMaxScaler(feature_range=(0, 1))
        df_val = scaler.fit_transform(df_val)
        df_train, df_test = df_val[0:size, :], df_val[(size-self.w_len):len(df_val), :]
        np.random.seed(7)
        return (df_train, df_test, scaler)
    
    def model(self):
        trainX, trainY = self.create_dataset(self.train, look_back=self.w_len)
        testX, testY = self.create_dataset(self.test, look_back=self.w_len)
        trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
        testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
        model = Sequential()
        model.add(LSTM(self.NFILTERS, input_shape=(1, self.w_len), return_sequences=True))
        model.add(LSTM(self.NFILTERS))
        model.add(Dense(1))
        model.compile(loss='mae', optimizer='adam')
        model.set_weights(weights = self.model_aux.get_weights()) 
        model.fit(trainX, trainY, epochs = self.NB_EPOCHS, batch_size=self.BATCH_SIZE, verbose=2, validation_split=self.val_split)
        predictions = list()
        for t in range(len(testX)):
            X = testX[t]
            yhat = model.predict(np.reshape(X, (1,1,self.w_len)))
            predictions.append(yhat[0])
        testPredict = self.scaler.inverse_transform([np.concatenate(predictions)])
        testY = np.hstack(testY)
        testY = self.scaler.inverse_transform([np.concatenate([np.expand_dims(i,axis=0) for i in testY])])
        mape = MAPE(testY[0], testPredict[0])
        return (mape)
    
class MV_LSTM_predictor(DataProcessing):
    def __init__(self, NFILTERS, BATCH_SIZE, NB_EPOCHS, w_len, val_split):
        DataProcessing.__init__(self)
        self.w_len = w_len
        self.train, self.test, self.scaler, self.size, self.df_val = self.train_test_split()
        self.NFILTERS = NFILTERS
        self.NB_EPOCHS = NB_EPOCHS
        self.BATCH_SIZE = BATCH_SIZE
        self.val_split = val_split
        self.mape = self.model()
        
    def series_to_supervised(self, data, n_in=1, n_out=1, dropnan=True):
        n_vars = 1 if type(data) is list else data.shape[1]
        df = pd.DataFrame(data)
        cols, names = list(), list()
        for i in range(n_in, 0, -1):
            cols.append(df.shift(i))
            names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
        for i in range(0, n_out):
            cols.append(df.shift(-i))
            if i == 0:
                names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
            else:
                names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
        agg = pd.concat(cols, axis=1)
        agg.columns = names
        if dropnan:
            agg.dropna(inplace=True)
        return agg
        
    def train_test_split(self): 
        df = self.lagged_df
        df = df[['Volume of mortgage loans', 'Russian Central Bank Key rate', 'Dollar/ruble exchange rate', 'Yandex Query 1']]
        size = int(len(df) * 0.7)
        df_val = df.values
        df_val = df_val.astype('float32')
        scaler = MinMaxScaler(feature_range=(0, 1))
        df_val = scaler.fit_transform(df_val)
        reframed = self.series_to_supervised(df_val, self.w_len, self.w_len)
        values = reframed.values
        train = values[0:size, :]
        test = values[(size-1):len(values), :]
        return (train, test, scaler, size, df_val)
    
    def model(self):
        train_X, train_y = self.train[:, [0, 1, 2, 3, 5, 6, 7]], self.train[:, 4]
        test_X, test_y = self.test[:, [0, 1, 2, 3, 5, 6, 7]], self.test[:, 4]
        train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
        test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
        model = Sequential()
        model.add(LSTM(self.NFILTERS, input_shape=(train_X.shape[1], train_X.shape[2]), return_sequences=True))
        model.add(LSTM(self.NFILTERS))
        model.add(Dense(1))
        model.compile(loss='mae', optimizer='adam')
        model.fit(train_X, train_y, epochs=self.NB_EPOCHS, batch_size=self.BATCH_SIZE, validation_split=self.val_split, verbose=2, shuffle=False)
        yhat = model.predict(test_X)
        check_test_X = self.df_val[self.size:]
        yhat = yhat.reshape((len(yhat), 1))
        inv_yhat = np.concatenate((yhat, check_test_X[:, 1:]), axis=1)
        inv_yhat = self.scaler.inverse_transform(inv_yhat)
        inv_yhat = inv_yhat[:,0]
        test_y = test_y.reshape((len(test_y), 1))
        inv_y = np.concatenate((test_y, check_test_X[:, 1:]), axis=1)
        inv_y = self.scaler.inverse_transform(inv_y)
        inv_y = inv_y[:,0]
        mape = MAPE(inv_y, inv_yhat)
        return (mape)