import optuna
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from src.environment.trading_environment import TradingEnv
from src.utils.hyperparam_config import suggest_hyperparams, TRAIN_TIMESTEPS_FOR_OPT, EVAL_EPISODES
import numpy as np

def objective(trial):
    """
    Objective function for Optuna hyperparameter optimization.
    Trains PPO model with suggested hyperparameters and returns average reward.
    """
    # Suggest hyperparameters using the config function
    params = suggest_hyperparams(trial)

    # Create environment
    env = TradingEnv({})

    # PPO model with suggested params
    model = PPO(
        'MlpPolicy',
        env,
        verbose=0,
        **params
    )

    # Train for a short period for quick evaluation
    model.learn(total_timesteps=TRAIN_TIMESTEPS_FOR_OPT)

    # Evaluate: Run a few episodes and get average reward
    rewards = []
    for episode in range(EVAL_EPISODES):
        obs, _ = env.reset()
        done = False
        episode_reward = 0
        while not done:
            action, _ = model.predict(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            done = terminated or truncated
        rewards.append(episode_reward)
        # Simple logging
        print(f"Trial {trial.number}, Episode {episode+1}: Reward={episode_reward}")

    avg_reward = np.mean(rewards)
    print(f"Trial {trial.number}: Avg Reward={avg_reward}\n")
    return avg_reward  # Maximize reward, so Optuna will maximize this

def run_optimization(env, n_trials=50, study_name='ppo_hyperparams'):
    """
    Run Optuna optimization for PPO hyperparameters with given environment.
    Returns best_params and study for visualization.
    """
    def objective(trial):
        params = suggest_hyperparams(trial)
        model = PPO('MlpPolicy', env, verbose=0, **params)
        model.learn(total_timesteps=TRAIN_TIMESTEPS_FOR_OPT)
        rewards = []
        for episode in range(EVAL_EPISODES):
            obs, _ = env.reset()
            done = False
            episode_reward = 0
            while not done:
                action, _ = model.predict(obs)
                obs, reward, terminated, truncated, info = env.step(action)
                episode_reward += reward
                done = terminated or truncated
            rewards.append(episode_reward)
            print(f"Trial {trial.number}, Episode {episode+1}: Reward={episode_reward}")
        avg_reward = np.mean(rewards)
        print(f"Trial {trial.number}: Avg Reward={avg_reward}\n")
        return avg_reward

    study = optuna.create_study(direction='maximize', study_name=study_name)
    study.optimize(objective, n_trials=n_trials)

    print("Best hyperparameters:")
    for key, value in study.best_params.items():
        print(f"{key}: {value}")
    print(f"Best reward: {study.best_value}")

    return study.best_params, study

if __name__ == "__main__":
    best_params = optimize_hyperparams()
    # Save best params to file or use in training
