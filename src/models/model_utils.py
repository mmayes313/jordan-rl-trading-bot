import joblib
from stable_baselines3 import PPO

def save_model(model, path):
    model.save(path)

def load_model(path):
    return PPO.load(path)
