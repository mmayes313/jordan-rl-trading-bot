import pandas as pd
import numpy as np
from src.indicators.technical_indicators import TechnicalIndicators

# Create dummy data with datetime index for resampling
dates = pd.date_range('2020-01-01', periods=1000, freq='1min')
data = pd.DataFrame({
    'open': np.random.rand(1000),
    'high': np.random.rand(1000) + 0.01,
    'low': np.random.rand(1000) - 0.01,
    'close': np.random.rand(1000),
    'volume': np.random.randint(100, 1000, 1000)
}, index=dates)

ti = TechnicalIndicators(data)
cci = ti.compute_cci()
sma = ti.compute_sma()
atr = ti.compute_atr()
adx = ti.compute_adx()
obv = ti.compute_obv()

print('CCI len:', len(cci))
print('SMA len:', len(sma))
print('ATR len:', len(atr))
print('ADX len:', len(adx))
print('OBV len:', len(obv))

total = len(np.concatenate([cci, sma, atr, adx, obv]))
print('Total indicators:', total)
print('Total ~372?', total >= 360 and total <= 380)
