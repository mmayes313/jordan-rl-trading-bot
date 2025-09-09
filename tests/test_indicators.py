import pytest
import pandas as pd
import numpy as np
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.indicators.technical_indicators import TechnicalIndicators

@pytest.fixture
def sample_data():
    """Create sample OHLCV data for testing"""
    np.random.seed(42)
    n = 100
    base_price = 1.10

    # Generate realistic price data
    returns = np.random.normal(0.0001, 0.01, n)
    prices = base_price * np.exp(np.cumsum(returns))

    # Create OHLCV data
    high_mult = 1 + np.random.uniform(0, 0.005, n)
    low_mult = 1 - np.random.uniform(0, 0.005, n)

    data = pd.DataFrame({
        'open': prices,
        'high': prices * high_mult,
        'low': prices * low_mult,
        'close': prices * (1 + np.random.normal(0, 0.002, n)),
        'volume': np.random.randint(1000, 10000, n)
    })

    # Ensure high >= max(open, close) and low <= min(open, close)
    data['high'] = np.maximum(data[['open', 'close']].max(axis=1), data['high'])
    data['low'] = np.minimum(data[['open', 'close']].min(axis=1), data['low'])

    # Add datetime index for resampling
    dates = pd.date_range('2024-01-01', periods=len(data), freq='1T')
    data.index = dates

    return data

def test_technical_indicators_initialization(sample_data):
    """Test TechnicalIndicators class initialization"""
    ti = TechnicalIndicators(sample_data)
    assert ti.data is not None
    assert len(ti.data) == len(sample_data)

def test_cci_computation(sample_data):
    """Test CCI computation returns correct shape"""
    ti = TechnicalIndicators(sample_data)
    cci = ti.compute_cci()

    # CCI should return 180 features (15 periods * 4 timeframes * 3 variations)
    assert len(cci) == 180
    assert isinstance(cci, np.ndarray)
    assert not np.isnan(cci).any()  # No NaN values

def test_sma_computation(sample_data):
    """Test SMA computation returns correct shape"""
    ti = TechnicalIndicators(sample_data)
    sma = ti.compute_sma()

    # SMA should return 168 features (7 periods * 4 timeframes * 6 variations)
    assert len(sma) == 168
    assert isinstance(sma, np.ndarray)
    assert not np.isnan(sma).any()

def test_atr_computation(sample_data):
    """Test ATR computation returns correct shape"""
    ti = TechnicalIndicators(sample_data)
    atr = ti.compute_atr()

    # ATR should return 16 features (4 periods * 4 timeframes)
    assert len(atr) == 16
    assert isinstance(atr, np.ndarray)
    assert not np.isnan(atr).any()

def test_adx_computation(sample_data):
    """Test ADX computation returns correct shape"""
    ti = TechnicalIndicators(sample_data)
    adx = ti.compute_adx()

    # ADX should return 4 features (1 period * 4 timeframes)
    assert len(adx) == 4
    assert isinstance(adx, np.ndarray)
    assert not np.isnan(adx).any()

def test_obv_computation(sample_data):
    """Test OBV computation returns correct shape"""
    ti = TechnicalIndicators(sample_data)
    obv = ti.compute_obv()

    # OBV should return 4 features (1 calculation * 4 timeframes)
    assert len(obv) == 4
    assert isinstance(obv, np.ndarray)
    assert not np.isnan(obv).any()

def test_total_features():
    """Test that total features equal 372"""
    # Create minimal data for quick test
    data = pd.DataFrame({
        'open': [1.0] * 50,
        'high': [1.01] * 50,
        'low': [0.99] * 50,
        'close': [1.0] * 50,
        'volume': [1000] * 50
    })

    ti = TechnicalIndicators(data)
    cci = ti.compute_cci()
    sma = ti.compute_sma()
    atr = ti.compute_atr()
    adx = ti.compute_adx()
    obv = ti.compute_obv()

    total_features = len(cci) + len(sma) + len(atr) + len(adx) + len(obv)
    assert total_features == 372

def test_indicator_values_range():
    """Test that indicator values are within reasonable ranges"""
    data = pd.DataFrame({
        'open': [1.0] * 100,
        'high': [1.02] * 100,
        'low': [0.98] * 100,
        'close': [1.01] * 100,
        'volume': [1000] * 100
    })

    ti = TechnicalIndicators(data)
    cci = ti.compute_cci()

    # CCI should typically be between -300 and +300
    assert np.all(cci >= -500) and np.all(cci <= 500)

def test_insufficient_data_handling():
    """Test behavior with insufficient data"""
    # Create data that's too short for some indicators
    data = pd.DataFrame({
        'open': [1.0] * 10,
        'high': [1.01] * 10,
        'low': [0.99] * 10,
        'close': [1.0] * 10,
        'volume': [1000] * 10
    })

    ti = TechnicalIndicators(data)

    # Should still work but may have NaN values that get handled
    cci = ti.compute_cci()
    assert len(cci) == 180  # Shape should still be correct
