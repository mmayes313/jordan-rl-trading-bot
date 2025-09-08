# Hyperparameter Optimization Guide

## Overview
This guide covers hyperparameter tuning for the PPO trading bot using Optuna Bayesian optimization.

## Step-by-Step Process

### 1. Load Data
```python
from src.data.data_loader import load_historical_data
data = load_historical_data('data/raw/eurusd_1y.csv')
```

### 2. Tune Hyperparameters
```bash
# Local quick test (20 trials)
python scripts/tune_hyperparams.py 20

# Colab full optimization (200 trials)
# Run notebooks/hyperparam_tuning.ipynb
```

### 3. Deploy Tuned Model
```python
from src.models.hyperparam_optimizer import run_optimization
from src.environment.historical_env import HistoricalTradingEnv

env = HistoricalTradingEnv(data)
best_params, study = run_optimization(env, n_trials=100)

# Train full model
from stable_baselines3 import PPO
model = PPO('MlpPolicy', env, **best_params)
model.learn(total_timesteps=1000000)
model.save('data/models/tuned_ppo_model')
```

## Pro Tips

- **Run Weekly**: Tune on fresh MT5 exports for adaptive trading
- **GPU Power**: Use Colab Pro for 200+ trials
- **Monitor Progress**: Check trial logs for convergence
- **Parameter Bounds**: Optimized for 417-feature trading environment

## Visualization
```python
import optuna.visualization as vis
vis.plot_param_importances(study).show()
vis.plot_optimization_history(study).show()
```

## Integration
- Dashboard Tab 3: View current hyperparams
- Jordan Chat: Daily optimization recommendations
- Model Persistence: Params saved alongside checkpoints

## Troubleshooting
- If freezing: Use `HistoricalTradingEnv` instead of `TradingEnv`
- Memory issues: Reduce `TRAIN_TIMESTEPS_FOR_OPT`
- Slow convergence: Increase trial count or adjust bounds

Happy optimizing! 🚀

## Best Practices

1. **Start Small**: Begin with 20-50 trials to get a feel for the search space
2. **Use GPU**: Enable GPU in Colab for faster optimization
3. **Monitor Progress**: Optuna provides visualization of the optimization process
4. **Validate Results**: Always test the tuned model on unseen data
5. **Iterate**: Run multiple studies and compare results

## Output

The tuning process outputs:
- Best hyperparameters found
- Optimization history and visualizations
- Tuned model saved to `data/models/tuned_ppo_model.zip`

## Troubleshooting

- **Slow Training**: Reduce `n_steps` or use GPU
- **Memory Issues**: Decrease `batch_size` or `n_steps`
- **Poor Performance**: Check environment setup and reward function
- **Optuna Errors**: Ensure all dependencies are installed

## Advanced Options

For distributed tuning, consider using Ray Tune instead of Optuna:
```bash
pip install ray[tune]
```

Then modify the optimizer to use Ray's API for parallel trials.
