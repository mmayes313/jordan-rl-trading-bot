import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.hyperparam_optimizer import run_optimization

if __name__ == "__main__":
    data_path = 'data/raw/eurusd_1y.csv'  # Update with real
    n_trials = int(sys.argv[1]) if len(sys.argv) > 1 else 50
    best = run_optimization(data_path, n_trials)
    print(f"Best params: {best}")
