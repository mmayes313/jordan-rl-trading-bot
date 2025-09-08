# Jordan RL Trading Bot Setup Guide

## Phase 1: Virtual Environment
- Create venv: `python -m venv venv`
- Activate: `venv\Scripts\activate` (Windows)
- Upgrade pip: `pip install --upgrade pip`

## Phase 2: Project Structure
- Create folders: config, src/environment, src/indicators, etc.
- Create __init__.py files in all directories
- Add .gitignore for venv, __pycache__, data/logs, etc.

## Phase 3: Dependencies
- Install from requirements.txt: `pip install -r requirements.txt`
- Note: Use torch==2.1.0 for CPU, or torch==2.1.0+cu118 for GPU

## Phase 4: Config Files
- trading_config.py: Daily target, DD, assets, timeframes
- model_config.py: PPO params
- dashboard_config.py: API keys

## Phase 5: Core Infrastructure
- MT5 connector, logger, data export script

## Phase 6: Technical Indicators
- CCI, SMA, ATR, ADX/OBV indicators

## Phase 7: Trading Environment
- Gym-compatible env with 417 obs space

## Phase 8: Reward System
- 21 rules for rewards

## Phase 9: PPO Model
- Train with stable-baselines3

## Phase 10: Streamlit Dashboard
- Live dashboard with tabs

## Phase 11: Jordan Personality
- API integration for chat

## Phase 12: Google Colab Integration
- Notebook for GPU training

## Phase 13: Testing Framework
- Pytest for all components

## Phase 14: Deployment & Commands
- Simple interface for train, live, dashboard
