# Jordan RL Trading Bot User Manual

## Overview
The Jordan RL Trading Bot is an autonomous trading system using PPO reinforcement learning for forex trading via MT5.

## Usage
1. Input target/DD in dashboard → Bot trades autonomously.
2. Chat with Jordan for insights via the dashboard chat tab.
3. Export data daily: `python scripts/simple_interface.py export`
4. Train in Colab: Upload notebook, run for GPU training.
5. Load model: Use `PPO.load('data/models/best_model')`

## Key Features
- Real-time dashboard with P&L, signals, model insights.
- Jordan personality chat for commentary and suggestions.
- Multi-asset, multi-timeframe support.
- Risk management with DD limits.

## Commands
- Train local: `python scripts/simple_interface.py train_local --days=100`
- Live trading: `python scripts/simple_interface.py live`
- Dashboard: `python scripts/simple_interface.py dashboard`
