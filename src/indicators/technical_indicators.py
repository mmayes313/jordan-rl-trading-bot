import pandas as pd
import talib
import numpy as np
from pandas_ta import adx, obv  # pip install pandas_ta if needed

class TechnicalIndicators:
    def __init__(self, data):
        self.data = data
        self.timeframes = ['1m', '15m', '1h', '1d']
        self.resampled = {tf: self.resample_data(tf) for tf in self.timeframes}

    def resample_data(self, tf):
        # Simple resample (use pandas resample for real)
        return self.data  # Placeholder; implement OHLCV resample

    def compute_cci(self, periods=[3,5,30,100,300]):
        features = []
        for tf_data in self.resampled.values():
            high, low, close = tf_data['high'], tf_data['low'], tf_data['close']
            for p in periods:
                cci_raw = talib.CCI(high, low, close, timeperiod=p)
                cci_sma2_shift2 = pd.Series(cci_raw).rolling(2).mean().shift(2)
                cci_sma5 = pd.Series(cci_raw).rolling(5).mean()
                features.extend([cci_raw, cci_sma2_shift2, cci_sma5])  # 15 base x 3 x 4 TF = 180
        return np.concatenate(features)

    def compute_sma(self, periods=range(1,21), additional=[50,200]):
        features = []
        for tf_data in self.resampled.values():
            close = tf_data['close']
            for p in periods:
                fwd_shift = pd.Series(close).rolling(p).mean().shift(-(p-1))  # Forward shift 0 to +19
                bwd_shift = pd.Series(close).rolling(p).mean().shift(p)  # Backward -1 to -20
                features.extend([fwd_shift, bwd_shift])
            features.append(pd.Series(close).rolling(50).mean())  # SMA50 shift=0
            features.append(pd.Series(close).rolling(200).mean())  # SMA200 shift=0
        return np.concatenate(features)  # 20 fwd + 20 bwd + 2 x 4 TF = 168

    def compute_atr(self, periods=[5,14]):
        features = []
        for tf_data in self.resampled.values():
            high, low, close = tf_data['high'], tf_data['low'], tf_data['close']
            for p in periods:
                atr_raw = talib.ATR(high, low, close, timeperiod=p)
                atr_sma2_shift2 = pd.Series(atr_raw).rolling(2).mean().shift(2)
                features.extend([atr_raw, atr_sma2_shift2])  # 2 p x 2 var x 4 TF = 16
        return np.concatenate(features)

    def compute_adx(self):
        features = []
        for tf_data in self.resampled.values():
            features.append(adx(tf_data['high'], tf_data['low'], tf_data['close'], length=14)['ADX_14'])  # 4 TF
        return np.concatenate(features)

    def compute_obv(self):
        features = []
        for tf_data in self.resampled.values():
            features.append(obv(tf_data['close'], tf_data['volume']))  # 4 TF
        return np.concatenate(features)
