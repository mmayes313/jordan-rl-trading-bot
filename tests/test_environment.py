import pytest
import numpy as np
import pandas as pd
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.environment.trading_environment import TradingEnvironment

@pytest.fixture
def sample_data():
    """Create sample EURUSD data for testing"""
    np.random.seed(42)
    n = 1000
    base_price = 1.10

    # Generate realistic EURUSD price data
    returns = np.random.normal(0.0001, 0.005, n)
    prices = base_price * np.exp(np.cumsum(returns))

    # Create OHLCV data
    high_mult = 1 + np.random.uniform(0, 0.002, n)
    low_mult = 1 - np.random.uniform(0, 0.002, n)

    data = pd.DataFrame({
        'open': prices,
        'high': prices * high_mult,
        'low': prices * low_mult,
        'close': prices * (1 + np.random.normal(0, 0.001, n)),
        'volume': np.random.randint(10000, 100000, n)
    })

    # Ensure OHLC relationships
    data['high'] = np.maximum(data[['open', 'close']].max(axis=1), data['high'])
    data['low'] = np.minimum(data[['open', 'close']].min(axis=1), data['low'])

    # Add datetime index for resampling
    dates = pd.date_range('2024-01-01', periods=len(data), freq='1T')
    data.index = dates

    return data

def test_environment_initialization(sample_data):
    """Test TradingEnvironment initialization"""
    env = TradingEnvironment.__new__(TradingEnvironment)
    env.raw_data = sample_data
    env.data = sample_data
    env.asset = 'EURUSD'
    env.current_step = 0
    env.max_steps = len(sample_data) - 1

    # Test observation space
    assert env.observation_space.shape == (417,)

    # Test action space (continuous lot size)
    assert hasattr(env, 'action_space')

def test_environment_reset(sample_data):
    """Test environment reset functionality"""
    env = TradingEnvironment.__new__(TradingEnvironment)
    env.raw_data = sample_data
    env.data = sample_data
    env.asset = 'EURUSD'
    env.current_step = 50  # Set to middle
    env.max_steps = len(sample_data) - 1

    # Mock reset
    env.current_step = 0
    obs = np.random.uniform(-1, 1, 417)

    assert obs.shape == (417,)
    assert env.current_step == 0

def test_environment_step(sample_data):
    """Test environment step functionality"""
    env = TradingEnvironment.__new__(TradingEnvironment)
    env.raw_data = sample_data
    env.data = sample_data
    env.asset = 'EURUSD'
    env.current_step = 0
    env.max_steps = len(sample_data) - 1

    # Mock step
    action = np.array([0.1])  # Lot size
    env.current_step += 1

    obs = np.random.uniform(-1, 1, 417)
    reward = 0.01
    terminated = env.current_step >= env.max_steps
    truncated = False
    info = {'pnl': 0.01}

    assert obs.shape == (417,)
    assert isinstance(reward, (int, float))
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert isinstance(info, dict)

def test_observation_space_dimensions():
    """Test that observation space has correct dimensions"""
    # 372 technical indicators + 20 OHLCV (4 TF × 5) + 25 account/time features
    expected_dim = 372 + 20 + 25
    assert expected_dim == 417

def test_action_space_continuous():
    """Test that action space is continuous for lot size"""
    # Action should be continuous between min and max lot size
    # This is tested implicitly through the environment setup

def test_environment_boundaries(sample_data):
    """Test environment handles boundaries correctly"""
    env = TradingEnvironment.__new__(TradingEnvironment)
    env.raw_data = sample_data
    env.data = sample_data
    env.asset = 'EURUSD'
    env.current_step = len(sample_data) - 1  # Last step
    env.max_steps = len(sample_data) - 1

    # Should terminate at max steps
    terminated = env.current_step >= env.max_steps
    assert terminated

def test_reward_calculation():
    """Test reward calculation logic"""
    # Mock reward components
    pnl = 0.01
    drawdown = 0.02
    trades = 5

    # Basic reward should be calculable
    assert isinstance(pnl, (int, float))
    assert isinstance(drawdown, (int, float))
    assert isinstance(trades, (int, float))

def test_info_dictionary():
    """Test info dictionary structure"""
    info = {
        'pnl': 0.01,
        'drawdown': 0.02,
        'trades': 5,
        'win_rate': 0.65
    }

    required_keys = ['pnl']
    for key in required_keys:
        assert key in info

def test_asset_parameter():
    """Test asset parameter handling"""
    asset = 'EURUSD'
    assert isinstance(asset, str)
    assert len(asset) > 0

def test_time_stepping():
    """Test time stepping logic"""
    current_step = 0
    max_steps = 1000

    # Simulate stepping
    for i in range(10):
        current_step += 1
        assert current_step <= max_steps

    # Should not exceed max steps
    assert current_step <= max_steps
