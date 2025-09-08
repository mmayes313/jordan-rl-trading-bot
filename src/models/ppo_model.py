from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import torch

def create_ppo_model(env, learning_rate=3e-4, n_steps=2048, batch_size=64, n_epochs=10,
                     gamma=0.99, gae_lambda=0.95, clip_range=0.2, ent_coef=0.0,
                     device='cuda' if torch.cuda.is_available() else 'cpu'):
    """
    Create a PPO model with specified hyperparameters.

    Args:
        env: Gym environment
        learning_rate: Learning rate for optimizer
        n_steps: Number of steps to collect before updating
        batch_size: Minibatch size for optimization
        n_epochs: Number of epochs for optimization
        gamma: Discount factor
        gae_lambda: GAE lambda parameter
        clip_range: PPO clip range
        ent_coef: Entropy coefficient
        device: Device to run on ('cuda' or 'cpu')

    Returns:
        PPO model ready for training
    """
    # Wrap environment if needed
    if not hasattr(env, 'num_envs'):
        env = DummyVecEnv([lambda: env])

    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        gamma=gamma,
        gae_lambda=gae_lambda,
        clip_range=clip_range,
        ent_coef=ent_coef,
        verbose=1,
        device=device
    )

    return model

def train_model(env, total_timesteps=1000000, device='cuda' if torch.cuda.is_available() else 'cpu'):
    model = PPO("MlpPolicy", env, verbose=1, device=device, learning_rate=3e-4)
    model.learn(total_timesteps=total_timesteps)
    model.save("data/models/best_model")
    return model

# Load: model = PPO.load("data/models/best_model")
