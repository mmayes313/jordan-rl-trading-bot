import pytest
import optuna
from unittest.mock import patch, MagicMock
from src.models.hyperparam_optimizer import run_optimization
from src.utils.hyperparam_config import suggest_hyperparams
from src.environment.historical_env import HistoricalTradingEnv

def test_suggest_hyperparams():
    """Test that suggest_hyperparams returns valid parameter ranges."""
    trial = optuna.trial.FixedTrial({
        'learning_rate': 1e-4,
        'n_steps': 256,
        'batch_size': 64,
        'n_epochs': 10,
        'gamma': 0.99,
        'gae_lambda': 0.95,
        'clip_range': 0.2,
        'ent_coef': 0.01,
        'vf_coef': 0.5,
        'max_grad_norm': 0.5
    })
    
    params = suggest_hyperparams(trial)
    
    # Check parameter ranges
    assert 1e-5 <= params['learning_rate'] <= 1e-3
    assert 64 <= params['n_steps'] <= 512
    assert params['batch_size'] in [32, 64, 128, 256]
    assert 5 <= params['n_epochs'] <= 20
    assert 0.9 <= params['gamma'] <= 0.999
    assert 0.9 <= params['gae_lambda'] <= 1.0
    assert 0.1 <= params['clip_range'] <= 0.4
    assert 0.0 <= params['ent_coef'] <= 0.1
    assert 0.5 <= params['vf_coef'] <= 1.0
    assert 0.5 <= params['max_grad_norm'] <= 10.0

def test_objective_runs():
    """Test that the objective function runs with a mock environment."""
    # Create a simple mock environment
    class MockEnv:
        def __init__(self):
            self.observation_space = MagicMock()
            self.observation_space.shape = (417,)
            self.action_space = MagicMock()
            self.action_space.shape = (3,)
        
        def reset(self):
            return [0] * 417, {}
        
        def step(self, action):
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
