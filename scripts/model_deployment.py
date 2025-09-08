from src.models.model_utils import load_model
from src.environment.mt5_connector import connect_mt5

model = load_model('data/models/ppo_model.zip')
if connect_mt5():
    obs = ...  # Get live obs
    action, _ = model.predict(obs)
    # Execute trade with action lot
    print("Deployed!")
