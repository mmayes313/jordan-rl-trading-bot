# Enhanced Training Output Documentation

## What You'll See While Training

When you run training with the new monitoring system, you'll see comprehensive real-time output including:

### 🤖 TRAINING PROGRESS
- **Timesteps**: Current training progress (e.g., 50,000/1,000,000)
- **Episodes**: Number of trading episodes completed
- **Runtime**: How long training has been running
- **Policy Loss**: Current policy gradient loss (should be non-zero and varying)
- **Value Loss**: Value function loss (should be non-zero and decreasing)
- **Entropy**: Exploration level (should start high, gradually decrease)
- **KL Divergence**: Policy change magnitude (should be > 0 but not too high)

### 💰 PERFORMANCE METRICS (Last 10 Episodes)
- **Avg Reward**: Average reward per episode (target: positive and increasing)
- **Avg PnL**: Average profit/loss in dollars per episode
- **Avg Drawdown**: Average maximum drawdown percentage
- **Avg Trades/Episode**: How many trades executed per episode
- **Win Rate**: Percentage of profitable trades (wins/total trades)

### 🎯 DAILY GOALS & RISK MANAGEMENT
- **Daily Target**: Your profit target (default: 1% daily)
- **Max Drawdown**: Risk limit (default: 5% max drawdown)
- **Goals Met**: How many episodes achieved the daily target
- **Risk Breaches**: How many episodes exceeded max drawdown

### 📈 CURRENT EPISODE STATUS
- **🟢/🔴 PnL**: Current episode profit/loss with status indicator
- **🟢/🟡/🔴 Drawdown**: Current drawdown with risk level indicator
- **Trades**: Number of trades in current episode
- **Wins**: Number of winning trades in current episode
- **Peak Equity**: Highest account balance reached

### 🏥 TRAINING HEALTH INDICATORS
- **🟢 Learning / 🔴 Stagnant**: Whether the model is actively learning
- **🟢 Exploring / 🟡 Low Exploration**: Whether the model is exploring new strategies
- **🟢 Active / 🔴 Inactive**: Whether trades are being executed

## Example Output

```
================================================================================
🤖 JORDAN RL TRADING BOT - TRAINING STATUS
================================================================================
📊 TRAINING PROGRESS:
   Timesteps: 25,000 | Episodes: 150 | Runtime: 0:15:30
   Policy Loss: -0.045000 | Value Loss: 0.025000
   Entropy: 0.750000 | KL Div: 0.002500

💰 PERFORMANCE METRICS (Last 10 Episodes):
   Avg Reward: +0.1250 | Avg PnL: $+15.75
   Avg Drawdown: 2.30% | Avg Trades/Episode: 8.5
   Win Rate: 62.5% (45/72 trades)

🎯 DAILY GOALS & RISK MANAGEMENT:
   Daily Target: 1.0% | Max Drawdown: 5.0%
   Goals Met: 23/150 (15.3%)
   Risk Breaches: 2/150 (1.3%)

📈 CURRENT EPISODE STATUS:
   🟢 PnL: $+12.50 | 🟢 Drawdown: 1.85%
   Trades: 6 | Wins: 4
   Peak Equity: $1,125.75

🏥 TRAINING HEALTH:
   🟢 Learning | 🟢 Exploring
   Trades per Episode: 🟢 Active
================================================================================
```

## Commands to Start Enhanced Training

### Local Training with Monitoring
```bash
python scripts/simple_interface.py train_local --days 10 --monitor-freq 100
```

### Demo Training (Short Run)
```bash
python scripts/demo_training.py
```

### Legacy Training (No Monitoring)
```bash
python scripts/simple_interface.py train_simple --days 10
```

## Files Created During Training

1. **`data/logs/training_metrics.json`** - Complete training metrics history
2. **`data/logs/diagnostic.log`** - Detailed diagnostic logs
3. **`data/models/best_model.zip`** - Trained PPO model
4. **`data/models/best_hyperparams.json`** - Optimized hyperparameters

## Interpreting the Output

### Good Signs (Training is Working)
- ✅ Policy Loss and Value Loss are non-zero and changing
- ✅ Win Rate > 50% and improving
- ✅ Avg PnL trending positive
- ✅ Risk Breaches < 5%
- ✅ Trades per Episode > 2 (model is active)

### Warning Signs (May Need Attention)
- ⚠️ Policy Loss stuck at 0 (no learning)
- ⚠️ Win Rate < 40% (poor strategy)
- ⚠️ High Risk Breaches > 10% (too aggressive)
- ⚠️ Trades per Episode < 1 (masks too restrictive)

### Critical Issues (Need Immediate Fix)
- 🚨 All losses = 0 (training stalled)
- 🚨 Avg PnL consistently negative
- 🚨 Risk Breaches > 20% (dangerous)
- 🚨 No trades being executed (environment issues)

The monitoring system will automatically identify these issues and provide specific fix recommendations in the logs.
