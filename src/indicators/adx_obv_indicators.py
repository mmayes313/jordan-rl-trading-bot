import talib
import pandas as pd

def calculate_adx(df, period):
    return talib.ADX(df['high'], df['low'], df['close'], timeperiod=period)

def calculate_obv(df):
    return talib.OBV(df['close'], df['volume'])
