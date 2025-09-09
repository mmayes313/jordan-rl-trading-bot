import pandas as pd
import talib
import numpy as np
# from pandas_ta import adx, obv  # Temporarily commented out due to compatibility issues

class TechnicalIndicators:
    def __init__(self, data):
        self.data = data
        self.timeframes = ['1min', '15min', '1H', '1D']  # Use pandas frequency strings
        self.resampled = {tf: self.resample_data(tf) for tf in self.timeframes}

    def resample_data(self, tf):
        # Resample data to specified timeframe
        return self.data.resample(tf).agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).dropna()

    def compute_cci(self, periods=[3,5,7,10,14,20,30,50,100,150,200,250,300,350,400]):
        features = []
        for tf_data in self.resampled.values():
            high, low, close = tf_data['high'], tf_data['low'], tf_data['close']
            for p in periods:
                cci_raw = talib.CCI(high, low, close, timeperiod=p)
                cci_sma2_shift2 = pd.Series(cci_raw).rolling(2).mean().shift(2)
                cci_sma5 = pd.Series(cci_raw).rolling(5).mean()
                # Convert to numpy arrays and handle NaN values
                features.extend([
                    np.nan_to_num(cci_raw.values, nan=0.0),
                    np.nan_to_num(cci_sma2_shift2.values, nan=0.0),
                    np.nan_to_num(cci_sma5.values, nan=0.0)
                ])
        return np.concatenate(features)

    def compute_sma(self, periods=range(1,21), additional=[50,200]):
        features = []
        for tf_data in self.resampled.values():
            close = tf_data['close']
            for p in periods:
                fwd_shift = pd.Series(close).rolling(p).mean().shift(-(p-1))  # Forward shift 0 to +19
                bwd_shift = pd.Series(close).rolling(p).mean().shift(p)  # Backward -1 to -20
                features.extend([
                    np.nan_to_num(fwd_shift.values, nan=0.0),
                    np.nan_to_num(bwd_shift.values, nan=0.0)
                ])
            # Add additional periods
            for p in additional:
                sma_val = pd.Series(close).rolling(p).mean()
                features.append(np.nan_to_num(sma_val.values, nan=0.0))
        return np.concatenate(features)

    def compute_atr(self, periods=[5,14]):
        features = []
        for tf_data in self.resampled.values():
            high, low, close = tf_data['high'], tf_data['low'], tf_data['close']
            for p in periods:
                atr_raw = talib.ATR(high, low, close, timeperiod=p)
                atr_sma2_shift2 = pd.Series(atr_raw).rolling(2).mean().shift(2)
                features.extend([
                    np.nan_to_num(atr_raw, nan=0.0),
                    np.nan_to_num(atr_sma2_shift2.values, nan=0.0)
                ])
        return np.concatenate(features)

    def compute_adx(self):
        features = []
        for tf_data in self.resampled.values():
            high, low, close = tf_data['high'], tf_data['low'], tf_data['close']
            # Use talib ADX instead of pandas_ta
            adx_val = talib.ADX(high, low, close, timeperiod=14)
            features.append(np.nan_to_num(adx_val, nan=50.0))
        return np.concatenate(features)

    def compute_obv(self):
        features = []
        for tf_data in self.resampled.values():
            close, volume = tf_data['close'], tf_data['volume']
            # Simple OBV implementation without pandas_ta
            obv_vals = np.zeros(len(close))
            for i in range(1, len(close)):
                if close.iloc[i] > close.iloc[i-1]:
                    obv_vals[i] = obv_vals[i-1] + volume.iloc[i]
                elif close.iloc[i] < close.iloc[i-1]:
                    obv_vals[i] = obv_vals[i-1] - volume.iloc[i]
                else:
                    obv_vals[i] = obv_vals[i-1]
            features.append(obv_vals)
        return np.concatenate(features)
