import pandas as pd
import numpy as np
from src.indicators.technical_indicators import TechnicalIndicators

# Create test data with datetime index
dates = pd.date_range('2024-01-01', periods=1000, freq='1T')
data = pd.DataFrame({
    'open': np.random.uniform(1.08, 1.12, 1000),
    'high': np.random.uniform(1.08, 1.12, 1000),
    'low': np.random.uniform(1.08, 1.12, 1000),
    'close': np.random.uniform(1.08, 1.12, 1000),
    'volume': np.random.randint(1000, 10000, 1000)
}, index=dates)

print(f"Data shape: {data.shape}")
print(f"Data index type: {type(data.index)}")
print(f"Data has datetime index: {isinstance(data.index, pd.DatetimeIndex)}")

try:
    ti = TechnicalIndicators(data)
    print("TechnicalIndicators initialized successfully")

    cci = ti.compute_cci()
    print(f"CCI computed successfully, length: {len(cci)}")

    sma = ti.compute_sma()
    print(f"SMA computed successfully, length: {len(sma)}")

    atr = ti.compute_atr()
    print(f"ATR computed successfully, length: {len(atr)}")

    adx = ti.compute_adx()
    print(f"ADX computed successfully, length: {len(adx)}")

    obv = ti.compute_obv()
    print(f"OBV computed successfully, length: {len(obv)}")

    total = len(np.concatenate([cci, sma, atr, adx, obv]))
    print(f"Total indicators: {total}")
    print(f"Target is ~372: {360 <= total <= 380}")

except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
    print("✅ gymnasium available")
except ImportError:
    print("❌ gymnasium not available")

print("\nBasic functionality test complete!")
