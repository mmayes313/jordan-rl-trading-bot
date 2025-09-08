import talib
import pandas as pd

def calculate_atr(df, period):
    return talib.ATR(df['high'], df['low'], df['close'], timeperiod=period)

def smooth_atr(atr_series, sma_period=2, shift=0):
    sma = talib.SMA(atr_series, timeperiod=sma_period)
    if shift > 0:
        return pd.Series(sma).shift(shift)
    elif shift < 0:
        return pd.Series(sma).shift(shift)
    return pd.Series(sma)
