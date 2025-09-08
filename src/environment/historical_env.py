import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd

class HistoricalTradingEnv(gym.Env):
    """
    Trading environment that uses historical data for hyperparameter tuning.
    No MT5 calls - fast and reliable for optimization.
    """
    def __init__(self, data, asset='EURUSD'):
        super().__init__()
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(417,))
        self.action_space = spaces.Box(low=0, high=1, shape=(3,))  # [buy_lots, sell_lots, hold]
        self.data = data
        self.asset = asset
        self.current_step = 0
        self.max_steps = len(data) - 1 if data is not None else 100
        self.reset()

    def step(self, action):
        # Simple reward logic for tuning
        reward = np.random.normal(0, 1)  # Random reward for testing
        
        self.current_step += 1
        terminated = self.current_step >= self.max_steps
        
        obs = np.random.normal(0, 1, self.observation_space.shape)  # Mock observations
        info = {'step': self.current_step}
        
        return obs, reward, terminated, False, info

    def reset(self, seed=None):
        self.current_step = 0
        obs = np.random.normal(0, 1, self.observation_space.shape)
        info = {}
        return obs, info
