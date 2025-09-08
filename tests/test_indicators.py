import pytest
from src.indicators.cci_indicators import calculate_cci, smooth_cci
import pandas as pd
import numpy as np

def test_calculate_cci():
    df = pd.DataFrame({
        'high': [1.1, 1.2, 1.3],
        'low': [0.9, 1.0, 1.1],
        'close': [1.0, 1.1, 1.2]
    })
    cci = calculate_cci(df, 14)
    assert len(cci) == 3
    # Add more assertions

def test_smooth_cci():
    cci_series = pd.Series([100, 110, 120])
    smoothed = smooth_cci(cci_series, 2, 0)
    assert len(smoothed) == 3
