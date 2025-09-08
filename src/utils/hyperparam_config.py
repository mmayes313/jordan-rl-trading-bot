"""
Hyperparameter configuration for PPO optimization.
Defines search spaces and default values.
"""

import optuna

HYPERPARAM_SEARCH_SPACE = {
    'learning_rate': {'type': 'float', 'low': 1e-5, 'high': 1e-3, 'log': True},
    'gamma': {'type': 'float', 'low': 0.9, 'high': 0.999},
    'gae_lambda': {'type': 'float', 'low': 0.9, 'high': 1.0},
    'clip_range': {'type': 'float', 'low': 0.1, 'high': 0.4},
    'ent_coef': {'type': 'float', 'low': 0.0, 'high': 0.1},
    'vf_coef': {'type': 'float', 'low': 0.5, 'high': 1.0},
    'max_grad_norm': {'type': 'float', 'low': 0.5, 'high': 10.0},
    'n_epochs': {'type': 'int', 'low': 5, 'high': 20},
    'batch_size': {'type': 'categorical', 'choices': [32, 64, 128, 256]},
    'n_steps': {'type': 'int', 'low': 64, 'high': 512},
}

DEFAULT_PPO_PARAMS = {
    'learning_rate': 3e-4,
    'gamma': 0.99,
    'gae_lambda': 0.95,
    'clip_range': 0.2,
    'ent_coef': 0.01,
    'vf_coef': 0.5,
    'max_grad_norm': 0.5,
    'n_epochs': 4,
    'batch_size': 64,
    'n_steps': 2048,
}

def suggest_hyperparams(trial):
    """
    Suggest hyperparameters for PPO optimization using Optuna.
    Tuned for 417-feature trading environment.
    """
    return {
        'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True),
        'n_steps': trial.suggest_int('n_steps', 64, 512),
        'batch_size': trial.suggest_categorical('batch_size', [32, 64, 128, 256]),
        'n_epochs': trial.suggest_int('n_epochs', 5, 20),
        'gamma': trial.suggest_float('gamma', 0.9, 0.999),
        'gae_lambda': trial.suggest_float('gae_lambda', 0.9, 1.0),
        'clip_range': trial.suggest_float('clip_range', 0.1, 0.4),
        'ent_coef': trial.suggest_float('ent_coef', 0.0, 0.1),
        'vf_coef': trial.suggest_float('vf_coef', 0.5, 1.0),
        'max_grad_norm': trial.suggest_float('max_grad_norm', 0.5, 10.0),
    }

# Evaluation settings (reduced for faster tuning)
EVAL_EPISODES = 2            # Fewer episodes for quicker evaluation
TRAIN_TIMESTEPS_FOR_OPT = 250  # Reduced steps for fast hyperparam tuning
FULL_TRAIN_TIMESTEPS = 100000  # Full training after optimization
