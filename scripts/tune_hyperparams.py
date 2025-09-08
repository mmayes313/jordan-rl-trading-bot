#!/usr/bin/env python
import sys
import json
import os
import subprocess
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

def run_setup():
    """Run the setup script to ensure dependencies are installed."""
    setup_path = Path(__file__).parent.parent / 'setup.py'
    print("🔧 Ensuring dependencies are installed...")
    try:
        result = subprocess.run([sys.executable, str(setup_path)],
                               check=True, capture_output=False, timeout=600)  # 10 minute timeout
        return True
    except subprocess.TimeoutExpired:
        print("⏰ Setup timed out after 10 minutes")
        return False
    except subprocess.CalledProcessError as e:
        print(f"❌ Setup failed: {e}")
        return False

# Run setup first
if not run_setup():
    print("❌ Cannot continue without proper setup")
    sys.exit(1)

from src.models.hyperparam_optimizer import run_optimization
from src.data.data_loader import load_historical_data
from src.environment.historical_env import HistoricalTradingEnv

if __name__ == "__main__":
    try:
        data = load_historical_data('data/raw/eurusd_1y.csv')  # Example
        print(f"Loaded {len(data)} data points")
    except Exception as e:
        print(f"Warning: Could not load data: {e}")
        data = None

    env = HistoricalTradingEnv(data, asset='EURUSD')
    n_trials = int(sys.argv[1]) if len(sys.argv) > 1 else 50

    print(f"Starting hyperparameter optimization with {n_trials} trials...")
    best_params, study = run_optimization(env, n_trials=n_trials)

    # Save to JSON
    os.makedirs('data/models', exist_ok=True)
    with open('data/models/best_hyperparams.json', 'w') as f:
        json.dump(best_params, f, indent=4)

    print(f"Tuned params saved to data/models/best_hyperparams.json: {best_params}")
