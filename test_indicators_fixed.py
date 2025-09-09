import pandas as pd
import numpy as np
from src.indicators.technical_indicators import TechnicalIndicators

# Create test data with more points for daily resampling
dates = pd.date_range('2024-01-01', periods=50000, freq='1T')  # 50k points = ~35 days
data = pd.DataFrame({
    'open': np.random.uniform(1.08, 1.12, 50000),
    'high': np.random.uniform(1.08, 1.12, 50000),
    'low': np.random.uniform(1.08, 1.12, 50000),
    'close': np.random.uniform(1.08, 1.12, 50000),
    'volume': np.random.randint(1000, 10000, 50000)
}, index=dates)

print('Data shape:', data.shape)
ti = TechnicalIndicators(data)

# Check resampled data sizes
for tf, df in ti.resampled.items():
    print(f'{tf}: {df.shape[0]} points')

# Test CCI computation
try:
    cci = ti.compute_cci()
    print('CCI length:', len(cci))
    print('CCI type:', type(cci).__name__)
    print('CCI first 5 values:', cci[:5] if len(cci) > 0 else 'Empty')
except Exception as e:
    print('CCI error:', e)
    import traceback
    traceback.print_exc()

# Test individual components
try:
    sma = ti.compute_sma()
    print('SMA length:', len(sma))
except Exception as e:
    print('SMA error:', e)

try:
    atr = ti.compute_atr()
    print('ATR length:', len(atr))
except Exception as e:
    print('ATR error:', e)

try:
    adx = ti.compute_adx()
    print('ADX length:', len(adx))
except Exception as e:
    print('ADX error:', e)

try:
    obv = ti.compute_obv()
    print('OBV length:', len(obv))
except Exception as e:
    print('OBV error:', e)

# Test concatenation
try:
    all_features = [cci, sma, atr, adx, obv]
    lengths = [len(arr) for arr in all_features]
    print('Individual lengths:', lengths)

    if all(lengths):
        total = np.concatenate(all_features)
        print('Total indicators:', len(total))
        print('SUCCESS!' if 360 <= len(total) <= 380 else 'ISSUE: Feature count mismatch')
    else:
        print('ERROR: Some arrays are empty!')
except Exception as e:
    print('Concatenation error:', e)
    import traceback
    traceback.print_exc()
