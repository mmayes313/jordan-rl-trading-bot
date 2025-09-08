import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces
from src.data.data_loader import load_historical_data
from src.data.data_processor import preprocess
from src.indicators.technical_indicators import TechnicalIndicators
from src.rewards.reward_system import RewardSystem
from src.masks.trading_masks import TradingMasks
from config.trading_config import TradingConfig

config = TradingConfig()

class TradingEnvironment(gym.Env):
    def __init__(self, data_path, asset='EURUSD'):
        super().__init__()
        self.raw_data = load_historical_data(data_path)
        self.data = self.raw_data  # Keep original for indexing
        self.asset = asset
        self.indicators = TechnicalIndicators(self.data)
        self.rewards = RewardSystem(config.daily_target)
        self.masks = TradingMasks(self.indicators)
        self.current_step = 0
        self.max_steps = len(self.data) - 1  # 1 trading day ~1440 for 1m
        # Obs space: 372 ind + 20 OHLCV (4TF x5) + 25 account/time
        obs_dim = 372 + 20 + 25
        self.observation_space = spaces.Box(low=-1, high=1, shape=(obs_dim,))
        self.action_space = spaces.Box(low=config.min_lot_size, high=config.max_lot_size, shape=(1,))  # Lot size continuous

    def reset(self, seed=None):
        super().reset(seed=seed)
        self.current_step = 0
        obs = self._get_obs()
        return obs, {}

    def step(self, action):
        lot_size = abs(action[0])  # Autonomous lot
        self.masks.apply_mask(self.indicators.compute_cci([30,100]))  # Check block
        reward = self.rewards.compute_reward(self._get_state(), action)
        self.current_step += 1
        terminated = self.current_step >= self.max_steps
        truncated = False
        obs = self._get_obs()
        info = {'pnl': self._get_pnl()}
        return obs, reward, terminated, truncated, info

    def _get_obs(self):
        # Mock 372 indicators (simplified for now)
        ind = np.random.uniform(-1, 1, 372)  # Mock indicators
        
        # Get current OHLCV data (mock 20 features: 4 timeframes * 5 OHLCV)
        ohlcv = np.array([1.1, 1.11, 1.09, 1.1, 1000] * 4)  # Mock OHLCV for 4 timeframes
        
        # Account features (11 features)
        account = np.array([config.initial_balance, 10000, 100, 9900, 0, 0, 0, 1.0, 0, 0, 0])
        
        # Time features (2 features)
        time_feat = np.array([self.current_step / 1440, 0])  # Normalized hour, day 0-6
        
        # Market conditions (4 features) 
        conditions = np.array([0.0001, 0.0002, 3600, 100])  # Spread, costs, time, leverage
        
        # Additional features to reach 25 total for account/time/conditions
        additional = np.zeros(8)  # 11 + 2 + 4 + 8 = 25
        
        full_obs = np.concatenate([ind, ohlcv, account, time_feat, conditions, additional])
        return full_obs

    def _get_state(self):
        return {
            'pnl': 0.01, 
            'drawdown': 0.02, 
            'trades': 5,
            'time': self.current_step * 60,  # Convert to seconds
            'assets': ['EURUSD'],
            'add_to_winner': False,
            'cci_entry': 50,
            'cci_exit': 50,
            'pullback': 0.3,
            'sma_cross': 0,
            'consec_fails': 0
        }

    def _get_pnl(self):
        return 0.01  # Mock

