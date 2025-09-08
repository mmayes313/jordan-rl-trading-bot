import talib
import pandas as pd

def calculate_cci(df, period):
    return talib.CCI(df['high'], df['low'], df['close'], timeperiod=period)

def smooth_cci(cci_series, sma_period=2, shift=0):
    sma = talib.SMA(cci_series, timeperiod=sma_period)
    if shift > 0:
        return pd.Series(sma).shift(shift)  # Forward
    elif shift < 0:
        return pd.Series(sma).shift(shift)  # Backward
    return pd.Series(sma)
