import pytest
import numpy as np
from src.environment.trading_environment import TradingEnv
from src.mt5_connector import get_all_symbols

def test_trading_env_init():
    config = {}  # Placeholder
    env = TradingEnv(config)
    assert env.observation_space.shape == (417,)
    assert env.action_space.shape == (3,)

def test_trading_env_step():
    config = {}
    env = TradingEnv(config)
    obs, info = env.reset()
    action = np.zeros(3)
    obs, reward, terminated, truncated, info = env.step(action)
    assert obs.shape == (417,)
    assert isinstance(reward, float)

def test_get_all_symbols():
    symbols = get_all_symbols()
    assert isinstance(symbols, list)
    # Note: This test may fail if MT5 is not running, but in a real setup it should return symbols
