import pytest
import json
import numpy as np
import sys
import os
from unittest.mock import patch, MagicMock

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.models.hyperparam_optimizer import run_optimization, objective
from src.utils.hyperparam_config import suggest_hyperparams

@pytest.fixture
def mock_trial():
    """Create mock trial for testing"""
    class MockTrial:
        def suggest_float(self, name, low, high, log=False):
            if name == 'learning_rate':
                return 0.0003
            elif name == 'gamma':
                return 0.99
            elif name == 'clip_range':
                return 0.2
            elif name == 'ent_coef':
                return 0.01
            return 0.5

        def suggest_int(self, name, low, high):
            if name == 'n_steps':
                return 2048
            elif name == 'n_epochs':
                return 10
            return 5

        def suggest_categorical(self, name, choices):
            if name == 'batch_size':
                return 64
            return choices[0]

    return MockTrial()

def test_suggest_hyperparams_structure(mock_trial):
    """Test that suggest_hyperparams returns correct parameter structure"""
    params = suggest_hyperparams(mock_trial)

    required_params = [
        'learning_rate', 'n_steps', 'batch_size', 'n_epochs',
        'gamma', 'clip_range', 'ent_coef'
    ]

    for param in required_params:
        assert param in params
        assert isinstance(params[param], (int, float))

def test_suggest_hyperparams_ranges(mock_trial):
    """Test that suggest_hyperparams returns parameters within valid ranges"""
    params = suggest_hyperparams(mock_trial)

    # Check learning rate range
    assert 1e-5 <= params['learning_rate'] <= 1e-3

    # Check n_steps range
    assert 64 <= params['n_steps'] <= 512

    # Check batch_size options
    assert params['batch_size'] in [32, 64, 128, 256]

    # Check n_epochs range
    assert 5 <= params['n_epochs'] <= 20

    # Check gamma range
    assert 0.9 <= params['gamma'] <= 0.999

    # Check clip_range range
    assert 0.1 <= params['clip_range'] <= 0.4

    # Check ent_coef range
    assert 0.0 <= params['ent_coef'] <= 0.1

def test_objective_function_basic():
    """Test that objective function runs without errors"""
    mock_trial = mock_trial()
    mock_env = MagicMock()

    # Mock PPO
    with patch('src.models.hyperparam_optimizer.PPO') as mock_ppo:
        mock_model = MagicMock()
        mock_ppo.return_value = mock_model

        result = objective(mock_trial, mock_env, timesteps=100)

        assert isinstance(result, (int, float))
        mock_model.learn.assert_called_once_with(total_timesteps=100)

@patch('src.models.hyperparam_optimizer.create_ppo_model')
@patch('src.models.hyperparam_optimizer.optuna.create_study')
def test_run_optimization_basic(mock_create_study, mock_create_model):
    """Test basic run_optimization functionality"""
    # Mock the model and study
    mock_model = MagicMock()
    mock_env = MagicMock()
    mock_model.get_env.return_value = mock_env
    mock_create_model.return_value = mock_model

    mock_study = MagicMock()
    mock_study.best_params = {'learning_rate': 0.0003, 'n_steps': 2048}
    mock_create_study.return_value = mock_study

    # Mock file operations
    with patch('builtins.open', create=True) as mock_open:
        mock_file = MagicMock()
        mock_open.return_value.__enter__.return_value = mock_file

        result = run_optimization('test_data.csv', n_trials=5)

        assert isinstance(result, dict)
        mock_create_study.assert_called_once_with(direction='maximize')
        mock_study.optimize.assert_called_once()
        mock_file.write.assert_called_once()

def test_run_optimization_file_creation():
    """Test that run_optimization creates the best_params.json file"""
    with patch('src.models.hyperparam_optimizer.create_ppo_model') as mock_create_model, \
         patch('src.models.hyperparam_optimizer.optuna.create_study') as mock_create_study, \
         patch('os.makedirs'), \
         patch('builtins.open', create=True) as mock_open:

        # Setup mocks
        mock_model = MagicMock()
        mock_env = MagicMock()
        mock_model.get_env.return_value = mock_env
        mock_create_model.return_value = mock_model

        mock_study = MagicMock()
        mock_study.best_params = {'learning_rate': 0.0003}
        mock_create_study.return_value = mock_study

        mock_file = MagicMock()
        mock_open.return_value.__enter__.return_value = mock_file

        # Run optimization
        result = run_optimization('test_data.csv', n_trials=5)

        # Verify file operations
        mock_open.assert_called_once()
        args, kwargs = mock_file.write.call_args
        written_data = args[0]

        # Should be valid JSON
        parsed_data = json.loads(written_data)
        assert isinstance(parsed_data, dict)

def test_run_optimization_with_different_trials():
    """Test run_optimization with different trial counts"""
    trial_counts = [1, 5, 10, 50]

    for n_trials in trial_counts:
        with patch('src.models.hyperparam_optimizer.create_ppo_model') as mock_create_model, \
             patch('src.models.hyperparam_optimizer.optuna.create_study') as mock_create_study, \
             patch('os.makedirs'), \
             patch('builtins.open', create=True):

            # Setup mocks
            mock_model = MagicMock()
            mock_env = MagicMock()
            mock_model.get_env.return_value = mock_env
            mock_create_model.return_value = mock_model

            mock_study = MagicMock()
            mock_study.best_params = {'learning_rate': 0.0003}
            mock_create_study.return_value = mock_study

            # Run optimization
            result = run_optimization('test_data.csv', n_trials=n_trials)

            assert isinstance(result, dict)

def test_objective_with_different_timesteps():
    """Test objective function with different timestep values"""
    timesteps = [1000, 10000, 50000, 100000]

    for ts in timesteps:
        mock_trial = mock_trial()
        mock_env = MagicMock()

        with patch('src.models.hyperparam_optimizer.PPO') as mock_ppo:
            mock_model = MagicMock()
            mock_ppo.return_value = mock_model

            result = objective(mock_trial, mock_env, timesteps=ts)

            assert isinstance(result, (int, float))
            mock_model.learn.assert_called_with(total_timesteps=ts)

def test_hyperparam_ranges_edge_cases():
    """Test hyperparameter ranges at edge cases"""
    # Test with extreme values (though suggest_hyperparams should clamp them)
    class ExtremeTrial:
        def suggest_float(self, name, low, high, log=False):
            if name == 'learning_rate':
                return 1e-6  # Below minimum
            elif name == 'gamma':
                return 0.8  # Below minimum
            elif name == 'clip_range':
                return 0.05  # Below minimum
            elif name == 'ent_coef':
                return -0.1  # Below minimum
            return 0.5

        def suggest_int(self, name, low, high):
            if name == 'n_steps':
                return 32  # Below minimum
            elif name == 'n_epochs':
                return 2  # Below minimum
            return 5

        def suggest_categorical(self, name, choices):
            return choices[0]

    extreme_trial = ExtremeTrial()
    params = suggest_hyperparams(extreme_trial)

    # Function should handle edge cases gracefully
    assert isinstance(params, dict)
    assert all(isinstance(v, (int, float)) for v in params.values())

def test_optimization_with_mock_data():
    """Test optimization with mock data path"""
    with patch('src.models.hyperparam_optimizer.create_ppo_model') as mock_create_model, \
         patch('src.models.hyperparam_optimizer.optuna.create_study') as mock_create_study, \
         patch('os.makedirs'), \
         patch('builtins.open', create=True):

        mock_model = MagicMock()
        mock_env = MagicMock()
        mock_model.get_env.return_value = mock_env
        mock_create_model.return_value = mock_model

        mock_study = MagicMock()
        mock_study.best_params = {
            'learning_rate': 0.0003,
            'n_steps': 2048,
            'batch_size': 64,
            'n_epochs': 10,
            'gamma': 0.99,
            'clip_range': 0.2,
            'ent_coef': 0.01
        }
        mock_create_study.return_value = mock_study

        result = run_optimization('mock_data.csv', n_trials=3)

        assert isinstance(result, dict)
        assert 'learning_rate' in result
        assert 'n_steps' in result

def test_best_params_json_structure():
    """Test that best_params.json has correct structure"""
    expected_keys = [
        'learning_rate', 'n_steps', 'batch_size', 'n_epochs',
        'gamma', 'clip_range', 'ent_coef'
    ]

    with patch('src.models.hyperparam_optimizer.create_ppo_model') as mock_create_model, \
         patch('src.models.hyperparam_optimizer.optuna.create_study') as mock_create_study, \
         patch('os.makedirs'), \
         patch('builtins.open', create=True) as mock_open:

        mock_model = MagicMock()
        mock_env = MagicMock()
        mock_model.get_env.return_value = mock_env
        mock_create_model.return_value = mock_model

        mock_study = MagicMock()
        best_params = {key: 0.5 for key in expected_keys}
        mock_study.best_params = best_params
        mock_create_study.return_value = mock_study

        mock_file = MagicMock()
        mock_open.return_value.__enter__.return_value = mock_file

        run_optimization('test.csv', n_trials=1)

        # Check what was written to file
        args, kwargs = mock_file.write.call_args
        written_json = args[0]
        parsed_params = json.loads(written_json)

        for key in expected_keys:
            assert key in parsed_params
            return [0] * 417, 1.0, True, False, {}

    env = MockEnv()
    
    trial = optuna.trial.FixedTrial({
        'learning_rate': 1e-4,
        'n_steps': 128,
        'batch_size': 64,
        'n_epochs': 5,
        'gamma': 0.99,
        'gae_lambda': 0.95,
        'clip_range': 0.2,
        'ent_coef': 0.01,
        'vf_coef': 0.5,
        'max_grad_norm': 0.5
    })

    with patch('src.models.hyperparam_optimizer.PPO') as mock_ppo:
        mock_model = MagicMock()
        mock_ppo.return_value = mock_model
        mock_model.learn.return_value = None
        mock_model.predict.return_value = ([0, 0, 0], None)

        # Import the objective function from the module
        from src.models.hyperparam_optimizer import run_optimization
        
        # Create a temporary objective function for testing
        def test_objective(trial):
            params = suggest_hyperparams(trial)
            model = mock_ppo('MlpPolicy', env, verbose=0, **params)
            model.learn(total_timesteps=100)  # Short for testing
            rewards = []
            for _ in range(2):  # Few episodes for testing
                obs, _ = env.reset()
                done = False
                episode_reward = 0
                steps = 0
                while not done and steps < 10:  # Limit steps
                    action, _ = model.predict(obs)
                    obs, reward, terminated, truncated, info = env.step(action)
                    episode_reward += reward
                    done = terminated or truncated
                    steps += 1
                rewards.append(episode_reward)
            return sum(rewards) / len(rewards) if rewards else 0

        score = test_objective(trial)
        assert isinstance(score, (int, float))

def test_run_optimization():
    """Test the run_optimization function with a mock environment."""
    env = HistoricalTradingEnv(None, 'EURUSD')
    
    with patch('optuna.create_study') as mock_study:
        mock_study_instance = MagicMock()
        mock_study.return_value = mock_study_instance
        mock_study_instance.best_params = {
            'learning_rate': 0.0001,
            'n_steps': 256,
            'batch_size': 64,
            'n_epochs': 10,
            'gamma': 0.99,
            'gae_lambda': 0.95,
            'clip_range': 0.2,
            'ent_coef': 0.01,
            'vf_coef': 0.5,
            'max_grad_norm': 0.5
        }
        mock_study_instance.best_value = 1.5

        best_params, study = run_optimization(env, n_trials=2)
        assert isinstance(best_params, dict)
        assert 'learning_rate' in best_params
        mock_study_instance.optimize.assert_called_once()

if __name__ == "__main__":
    pytest.main([__file__])
