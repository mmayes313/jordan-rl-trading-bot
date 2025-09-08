import talib
import pandas as pd

def calculate_shifted_sma(close_prices, period, shift):
    sma = talib.SMA(close_prices, timeperiod=period)
    return pd.Series(sma).shift(shift)
# shifts dict: {'1':0, '2':1, ..., '20':19} forward; {'1':-1, ..., '20':-20} backward
