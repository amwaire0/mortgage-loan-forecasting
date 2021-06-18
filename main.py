from models import Baseline_predictor, ARIMA_predictor, ARIMAX_predictor, LSTM_predictor, Transfer_learning_predictor, MV_LSTM_predictor

class MortgageVolumePrediction:
    __arima_params = {'p': 1,
                      'd': 0,
                      'q': 1}
    __arimax_params = {'p': 1,
                      'd': 0,
                      'q': 1}
    __lstm_params = {'number of filters': 500,
                     'batch size': 72,
                     'number of epochs': 50,
                     'window length': 1,
                     'validation split': 0.3}
    __transfer_params = {'number of filters': 500,
                     'batch size': 72,
                     'number of epochs': 50,
                     'window length': 1,
                     'validation split': 0.3}
    __mv_lstm_params = {'number of filters': 500,
                     'batch size': 72,
                     'number of epochs': 50,
                     'window length': 1,
                     'validation split': 0.3}
        
    def set_ARIMA_model(self, p = 1, d = 0, q = 1):
        self.__arima_params['p'] = p
        self.__arima_params['d'] = d
        self.__arima_params['q'] = q
        
    def set_ARIMAX_model(self, p = 1, d = 0, q = 1):
        self.__arimax_params['p'] = p
        self.__arimax_params['d'] = d
        self.__arimax_params['q'] = q
        
    def set_LSTM_model(self, NFILTERS = 500, BATCH_SIZE = 72, NB_EPOCHS = 50, w_len = 1, val_split = 0.3):
        self.__lstm_params['number of filters'] = NFILTERS
        self.__lstm_params['batch size'] = BATCH_SIZE
        self.__lstm_params['number of epochs'] = NB_EPOCHS
        self.__lstm_params['window length'] = w_len
        self.__lstm_params['validation split'] = val_split
        
    def set_transfer_model(self, NFILTERS = 500, BATCH_SIZE = 72, NB_EPOCHS = 50, w_len = 1, val_split = 0.3):
        self.__transfer_params['number of filters'] = NFILTERS
        self.__transfer_params['batch size'] = BATCH_SIZE
        self.__transfer_params['number of epochs'] = NB_EPOCHS
        self.__transfer_params['window length'] = w_len
        self.__transfer_params['validation split'] = val_split    
        
    def set_MV_LSTM_model(self, NFILTERS = 500, BATCH_SIZE = 72, NB_EPOCHS = 50, w_len = 1, val_split = 0.3):
        self.__mv_lstm_params['number of filters'] = NFILTERS
        self.__mv_lstm_params['batch size'] = BATCH_SIZE
        self.__mv_lstm_params['number of epochs'] = NB_EPOCHS
        self.__mv_lstm_params['window length'] = w_len
        self.__mv_lstm_params['validation split'] = val_split
        
    def infer(self):
        model_b = Baseline_predictor()
        model_arima = ARIMA_predictor(self.__arima_params['p'], self.__arima_params['d'], self.__arima_params['q'])
        model_arimax = ARIMAX_predictor(self.__arimax_params['p'], self.__arimax_params['d'], self.__arimax_params['q'])
        model_lstm = LSTM_predictor(self.__lstm_params['number of filters'], self.__lstm_params['batch size'], self.__lstm_params['number of epochs'], 
                                    self.__lstm_params['window length'], self.__lstm_params['validation split'])
        model_transfer = Transfer_learning_predictor(self.__transfer_params['number of filters'], self.__transfer_params['batch size'],
                                                     self.__transfer_params['number of epochs'], self.__transfer_params['window length'], 
                                                     self.__transfer_params['validation split'])
        model_mv_lstm = MV_LSTM_predictor(self.__mv_lstm_params['number of filters'], self.__mv_lstm_params['batch size'], 
                                    self.__mv_lstm_params['number of epochs'], self.__mv_lstm_params['window length'], self.__mv_lstm_params['validation split'])
        mape_b = model_b.mape
        mape_a = model_arima.mape
        mape_ax = model_arimax.mape
        mape_l = model_lstm.mape
        mape_t = model_transfer.mape
        mape_ml = model_mv_lstm.mape
        return mape_b, mape_a, mape_ax, mape_l, mape_t, mape_ml

MVP = MortgageVolumePrediction()
MVP.set_ARIMA_model(p = 1, d = 0, q = 1)
MVP.set_ARIMAX_model(p = 1, d = 0, q = 1)
MVP.set_LSTM_model(NFILTERS = 500, BATCH_SIZE = 72, NB_EPOCHS = 50, w_len = 1, val_split = 0.3)
MVP.set_transfer_model(NFILTERS = 500, BATCH_SIZE = 72, NB_EPOCHS = 50, w_len = 1, val_split = 0.3)
MVP.set_MV_LSTM_model(NFILTERS = 500, BATCH_SIZE = 72, NB_EPOCHS = 50, w_len = 1, val_split = 0.3)
mape_b, mape_a, mape_ax, mape_l, mape_t, mape_ml = MVP.infer()