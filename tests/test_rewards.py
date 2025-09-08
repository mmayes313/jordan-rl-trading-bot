import pytest
from src.rewards.reward_system import RewardSystem

def test_reward_system():
    reward_sys = RewardSystem()
    state = {'time_of_day': 0.5, 'equity': 1000, 'max_dd': 0.02, 'pnl': 50}
    action = 'buy'
    pnl_change = 10
    reward = reward_sys.get_reward(state, action, pnl_change, is_end_of_episode=False)
    assert isinstance(reward, float)
    # Add more assertions for rules
