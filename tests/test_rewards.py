import pytest
import numpy as np
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.rewards.reward_system import RewardSystem

@pytest.fixture
def reward_system():
    """Create RewardSystem instance for testing"""
    return RewardSystem(daily_target=0.03)

def test_reward_system_initialization(reward_system):
    """Test RewardSystem initialization"""
    assert reward_system.daily_target == 0.03
    assert hasattr(reward_system, 'compute_reward')

def test_reward_computation_basic(reward_system):
    """Test basic reward computation"""
    state = {
        'pnl': 0.02,  # Below target
        'drawdown': 0.01,
        'trades': 5,
        'time': 3600,  # 1 hour
        'assets': ['EURUSD'],
        'add_to_winner': False,
        'cci_entry': 50,
        'cci_exit': 50,
        'pullback': 0.3,
        'sma_cross': 0,
        'consec_fails': 0
    }

    action = np.array([0.1])
    reward = reward_system.compute_reward(state, action)

    assert isinstance(reward, (int, float))
    assert reward >= -5 and reward <= 5  # Reasonable reward range

def test_daily_target_reward(reward_system):
    """Test daily target reward calculation"""
    # Test achieving target
    state_target = {
        'pnl': 0.04,  # Above target
        'drawdown': 0.01,
        'trades': 5,
        'time': 3600,
        'assets': ['EURUSD'],
        'add_to_winner': False,
        'cci_entry': 50,
        'cci_exit': 50,
        'pullback': 0.3,
        'sma_cross': 0,
        'consec_fails': 0
    }

    reward_above = reward_system.compute_reward(state_target, np.array([0.1]))

    # Test below target
    state_below = state_target.copy()
    state_below['pnl'] = 0.01

    reward_below = reward_system.compute_reward(state_below, np.array([0.1]))

    assert reward_above > reward_below

def test_time_based_bonus(reward_system):
    """Test time-based bonus for quick target achievement"""
    state_quick = {
        'pnl': 0.04,
        'drawdown': 0.01,
        'trades': 5,
        'time': 7200,  # 2 hours - quick achievement
        'assets': ['EURUSD'],
        'add_to_winner': False,
        'cci_entry': 50,
        'cci_exit': 50,
        'pullback': 0.3,
        'sma_cross': 0,
        'consec_fails': 0
    }

    state_slow = state_quick.copy()
    state_slow['time'] = 14400  # 4 hours - slower achievement

    reward_quick = reward_system.compute_reward(state_quick, np.array([0.1]))
    reward_slow = reward_system.compute_reward(state_slow, np.array([0.1]))

    assert reward_quick > reward_slow

def test_drawdown_penalty(reward_system):
    """Test drawdown penalty"""
    state_low_dd = {
        'pnl': 0.02,
        'drawdown': 0.02,  # Low drawdown
        'trades': 5,
        'time': 3600,
        'assets': ['EURUSD'],
        'add_to_winner': False,
        'cci_entry': 50,
        'cci_exit': 50,
        'pullback': 0.3,
        'sma_cross': 0,
        'consec_fails': 0
    }

    state_high_dd = state_low_dd.copy()
    state_high_dd['drawdown'] = 0.08  # High drawdown

    reward_low_dd = reward_system.compute_reward(state_low_dd, np.array([0.1]))
    reward_high_dd = reward_system.compute_reward(state_high_dd, np.array([0.1]))

    assert reward_low_dd > reward_high_dd

def test_trade_efficiency_bonus(reward_system):
    """Test trade efficiency bonus"""
    state_efficient = {
        'pnl': 0.02,
        'drawdown': 0.01,
        'trades': 5,
        'time': 3600,
        'assets': ['EURUSD'],
        'add_to_winner': False,
        'cci_entry': 50,
        'cci_exit': 50,
        'pullback': 0.3,
        'sma_cross': 0,
        'consec_fails': 0
    }

    state_inefficient = state_efficient.copy()
    state_inefficient['trades'] = 20  # More trades for same P&L

    reward_efficient = reward_system.compute_reward(state_efficient, np.array([0.1]))
    reward_inefficient = reward_system.compute_reward(state_inefficient, np.array([0.1]))

    assert reward_efficient > reward_inefficient

def test_multiple_assets_bonus(reward_system):
    """Test multiple assets bonus"""
    state_single = {
        'pnl': 0.02,
        'drawdown': 0.01,
        'trades': 5,
        'time': 3600,
        'assets': ['EURUSD'],
        'add_to_winner': False,
        'cci_entry': 50,
        'cci_exit': 50,
        'pullback': 0.3,
        'sma_cross': 0,
        'consec_fails': 0
    }

    state_multi = state_single.copy()
    state_multi['assets'] = ['EURUSD', 'GBPUSD', 'USDJPY']

    reward_single = reward_system.compute_reward(state_single, np.array([0.1]))
    reward_multi = reward_system.compute_reward(state_multi, np.array([0.1]))

    assert reward_multi > reward_single

def test_add_to_winner_bonus(reward_system):
    """Test add to winner bonus"""
    state_no_add = {
        'pnl': 0.02,
        'drawdown': 0.01,
        'trades': 5,
        'time': 3600,
        'assets': ['EURUSD'],
        'add_to_winner': False,
        'cci_entry': 50,
        'cci_exit': 50,
        'pullback': 0.3,
        'sma_cross': 0,
        'consec_fails': 0
    }

    state_add = state_no_add.copy()
    state_add['add_to_winner'] = True

    reward_no_add = reward_system.compute_reward(state_no_add, np.array([0.1]))
    reward_add = reward_system.compute_reward(state_add, np.array([0.1]))

    assert reward_add > reward_no_add

def test_cci_quality_rewards(reward_system):
    """Test CCI entry and exit quality rewards"""
    state_good_cci = {
        'pnl': 0.02,
        'drawdown': 0.01,
        'trades': 5,
        'time': 3600,
        'assets': ['EURUSD'],
        'add_to_winner': False,
        'cci_entry': -120,  # Good oversold entry
        'cci_exit': 80,     # Good exit timing
        'pullback': 0.3,
        'sma_cross': 0,
        'consec_fails': 0
    }

    state_poor_cci = state_good_cci.copy()
    state_poor_cci['cci_entry'] = 20  # Poor entry

    reward_good_cci = reward_system.compute_reward(state_good_cci, np.array([0.1]))
    reward_poor_cci = reward_system.compute_reward(state_poor_cci, np.array([0.1]))

    assert reward_good_cci > reward_poor_cci

def test_consecutive_failures_penalty(reward_system):
    """Test consecutive failures penalty"""
    state_no_fails = {
        'pnl': 0.02,
        'drawdown': 0.01,
        'trades': 5,
        'time': 3600,
        'assets': ['EURUSD'],
        'add_to_winner': False,
        'cci_entry': 50,
        'cci_exit': 50,
        'pullback': 0.3,
        'sma_cross': 0,
        'consec_fails': 0
    }

    state_many_fails = state_no_fails.copy()
    state_many_fails['consec_fails'] = 5

    reward_no_fails = reward_system.compute_reward(state_no_fails, np.array([0.1]))
    reward_many_fails = reward_system.compute_reward(state_many_fails, np.array([0.1]))

    assert reward_no_fails > reward_many_fails

def test_action_size_penalty(reward_system):
    """Test action size penalty for oversized positions"""
    state_normal = {
        'pnl': 0.02,
        'drawdown': 0.01,
        'trades': 5,
        'time': 3600,
        'assets': ['EURUSD'],
        'add_to_winner': False,
        'cci_entry': 50,
        'cci_exit': 50,
        'pullback': 0.3,
        'sma_cross': 0,
        'consec_fails': 0
    }

    action_normal = np.array([0.5])  # Normal lot size
    action_large = np.array([2.0])   # Oversized lot

    reward_normal = reward_system.compute_reward(state_normal, action_normal)
    reward_large = reward_system.compute_reward(state_normal, action_large)

    assert reward_normal > reward_large

def test_perfect_day_bonus(reward_system):
    """Test perfect day bonus"""
    state_perfect = {
        'pnl': 0.04,      # Above target
        'drawdown': 0.01, # Low drawdown
        'trades': 15,     # Reasonable trade count
        'time': 3600,
        'assets': ['EURUSD'],
        'add_to_winner': False,
        'cci_entry': 50,
        'cci_exit': 50,
        'pullback': 0.3,
        'sma_cross': 0,
        'consec_fails': 0
    }

    state_imperfect = state_perfect.copy()
    state_imperfect['drawdown'] = 0.05  # Higher drawdown

    reward_perfect = reward_system.compute_reward(state_perfect, np.array([0.1]))
    reward_imperfect = reward_system.compute_reward(state_imperfect, np.array([0.1]))

    assert reward_perfect > reward_imperfect

def test_reward_range_bounds(reward_system):
    """Test that rewards stay within reasonable bounds"""
    test_states = [
        {'pnl': 0.10, 'drawdown': 0.01, 'trades': 5, 'time': 3600, 'assets': ['EURUSD'],
         'add_to_winner': True, 'cci_entry': -150, 'cci_exit': 100, 'pullback': 0.3,
         'sma_cross': 1, 'consec_fails': 0},  # Very good state
        {'pnl': -0.10, 'drawdown': 0.15, 'trades': 50, 'time': 3600, 'assets': ['EURUSD'],
         'add_to_winner': False, 'cci_entry': 50, 'cci_exit': 50, 'pullback': 0.1,
         'sma_cross': 0, 'consec_fails': 10}  # Very bad state
    ]

    for state in test_states:
        reward = reward_system.compute_reward(state, np.array([0.1]))
        assert isinstance(reward, (int, float))
        # Allow reasonable range for extreme cases
        assert reward >= -10 and reward <= 10
