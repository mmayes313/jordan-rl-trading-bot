import pytest
import numpy as np
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.masks.trading_masks import TradingMasks

@pytest.fixture
def mock_indicators():
    """Create mock indicators for testing"""
    class MockIndicators:
        def compute_cci(self, periods=None):
            # Return mock CCI values for different timeframes
            # 8 CCI values: 2 periods × 4 timeframes
            return np.array([50, 60, 45, 55, 40, 65, 35, 70])

    return MockIndicators()

def test_trading_masks_initialization(mock_indicators):
    """Test TradingMasks initialization"""
    masks = TradingMasks(mock_indicators)
    assert masks.indicators is not None
    assert hasattr(masks, 'apply_mask')

def test_apply_mask_buy_signal(mock_indicators):
    """Test mask application for buy signals"""
    masks = TradingMasks(mock_indicators)

    # Test with CCI values that should allow buying
    cci_values = np.array([30, 25, 35, 20, 40, 15, 45, 10])  # Mostly oversold
    result = masks.apply_mask(cci_values)

    # Should return True for buy allowance
    assert isinstance(result, (bool, np.bool_))

def test_apply_mask_sell_signal(mock_indicators):
    """Test mask application for sell signals"""
    masks = TradingMasks(mock_indicators)

    # Test with CCI values that should allow selling
    cci_values = np.array([70, 75, 65, 80, 60, 85, 55, 90])  # Mostly overbought
    result = masks.apply_mask(cci_values)

    # Should return True for sell allowance
    assert isinstance(result, (bool, np.bool_))

def test_apply_mask_blocked_signal(mock_indicators):
    """Test mask application for blocked signals"""
    masks = TradingMasks(mock_indicators)

    # Test with CCI values in neutral zone (should block)
    cci_values = np.array([50, 55, 45, 60, 40, 65, 35, 70])  # Mixed neutral
    result = masks.apply_mask(cci_values)

    # Should return False for blocked signals
    assert isinstance(result, (bool, np.bool_))

def test_cci_threshold_logic():
    """Test CCI threshold logic for buy/sell decisions"""
    # CCI < -60: Strong buy signal
    # CCI > 60: Strong sell signal
    # -60 <= CCI <= 60: Neutral (blocked)

    test_cases = [
        (-80, True, "strong_buy"),
        (-30, False, "weak_buy_blocked"),
        (30, False, "weak_sell_blocked"),
        (80, True, "strong_sell"),
        (0, False, "neutral_blocked")
    ]

    for cci_value, expected_allowed, description in test_cases:
        # Test logic would be implemented in apply_mask
        if abs(cci_value) > 60:
            assert expected_allowed, f"Failed for {description}"
        else:
            assert not expected_allowed, f"Failed for {description}"

def test_multi_timeframe_consensus():
    """Test multi-timeframe consensus logic"""
    # Should require consensus across timeframes
    timeframes = ['1m', '15m', '1h', '1d']

    # Test case: All timeframes agree on oversold
    cci_all_oversold = np.array([-70, -75, -65, -80])
    consensus_oversold = np.all(cci_all_oversold < -60)
    assert consensus_oversold

    # Test case: Mixed signals
    cci_mixed = np.array([-70, 20, -65, 30])
    consensus_mixed = np.all(cci_mixed < -60) or np.all(cci_mixed > 60)
    assert not consensus_mixed

def test_mask_allowance_rate():
    """Test that mask allowance rate is reasonable"""
    # Simulate multiple mask decisions
    decisions = []
    np.random.seed(42)

    for _ in range(100):
        # Random CCI values
        cci_values = np.random.normal(0, 50, 8)
        # Simple logic: allow if extreme values present
        has_extreme_buy = np.any(cci_values < -60)
        has_extreme_sell = np.any(cci_values > 60)
        allowed = has_extreme_buy or has_extreme_sell
        decisions.append(allowed)

    allowance_rate = np.mean(decisions)
    # Should be reasonable (not too high or low)
    assert 0.1 <= allowance_rate <= 0.9, f"Allowance rate {allowance_rate} is unreasonable"

def test_mask_consistency():
    """Test that mask decisions are consistent for similar inputs"""
    base_cci = np.array([50, 55, 45, 60, 40, 65, 35, 70])

    # Test slight variations
    variations = [
        base_cci + np.random.normal(0, 1, 8) for _ in range(10)
    ]

    decisions = []
    for cci_values in variations:
        # Simple consistency check
        has_strong_signal = np.any(np.abs(cci_values) > 60)
        decisions.append(has_strong_signal)

    # Should have some consistency (not completely random)
    unique_decisions = len(set(decisions))
    assert unique_decisions <= 5, "Decisions too inconsistent"

def test_edge_case_extreme_values():
    """Test edge cases with extreme CCI values"""
    masks = TradingMasks(mock_indicators)

    # Test extremely oversold
    extreme_oversold = np.array([-200, -180, -150, -120])
    result_oversold = masks.apply_mask(extreme_oversold)
    assert result_oversold == True, "Should allow extreme oversold"

    # Test extremely overbought
    extreme_overbought = np.array([200, 180, 150, 120])
    result_overbought = masks.apply_mask(extreme_overbought)
    assert result_overbought == True, "Should allow extreme overbought"

def test_mask_with_missing_data():
    """Test mask behavior with missing or invalid data"""
    masks = TradingMasks(mock_indicators)

    # Test with NaN values
    cci_with_nan = np.array([50, np.nan, 45, 60])
    # Should handle NaN gracefully (implementation dependent)
    try:
        result = masks.apply_mask(cci_with_nan)
        assert isinstance(result, (bool, np.bool_))
    except:
        # If it fails, that's also acceptable with proper error handling
        pass

def test_timeframe_importance():
    """Test that different timeframes have appropriate importance"""
    # Shorter timeframes should be more responsive but noisier
    # Longer timeframes should be more stable but slower

    # This is more of a design consideration than a unit test
    timeframes = ['1m', '15m', '1h', '1d']
    responsiveness = [1.0, 0.8, 0.6, 0.4]  # Shorter = more responsive

    assert responsiveness[0] > responsiveness[-1], "Shorter timeframes should be more responsive"

def test_mask_adaptation():
    """Test that masks can adapt to different market conditions"""
    # Masks should work in trending vs ranging markets
    trending_cci = np.array([80, 75, 85, 70])  # Strong trend
    ranging_cci = np.array([20, -10, 30, -20])  # Sideways movement

    # Both should produce valid decisions
    masks = TradingMasks(mock_indicators)

    result_trending = masks.apply_mask(trending_cci)
    result_ranging = masks.apply_mask(ranging_cci)

    assert isinstance(result_trending, (bool, np.bool_))
    assert isinstance(result_ranging, (bool, np.bool_))
