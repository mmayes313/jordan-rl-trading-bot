from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import torch

def train_model(env, total_timesteps=1000000, device='cuda' if torch.cuda.is_available() else 'cpu'):
    model = PPO("MlpPolicy", env, verbose=1, device=device, learning_rate=3e-4)
    model.learn(total_timesteps=total_timesteps)
    model.save("data/models/best_model")
    return model

# Load: model = PPO.load("data/models/best_model")
