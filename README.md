# Jordan RL Trading Bot

A sophisticated reinforcement learning trading bot inspired by Jordan Belfort, featuring PPO optimization, MT5 integration, and real-time dashboard.

## 🚀 Project Phases

### Phase 1: Core Infrastructure
- ✅ Virtual environment setup
- ✅ Project structure with modular src/
- ✅ Dependencies management (requirements.txt)
- ✅ Basic logging and configuration

### Phase 2: Data Pipeline
- ✅ MT5 connector for live data
- ✅ Historical data export (OHLCV)
- ✅ Technical indicators (CCI, SMA, ATR, ADX, OBV)
- ✅ Data preprocessing and feature engineering

### Phase 3: Trading Environment
- ✅ Gymnasium-based TradingEnv (417 obs space)
- ✅ Action spaces for buy/sell/hold with lot sizes
- ✅ Reward system with 21 configurable rules
- ✅ Trailing drawdown tracking via MT5

### Phase 4: Reward System
- ✅ 21 reward rules implementation
- ✅ Dynamic configuration from config files
- ✅ Risk management integration
- ✅ Performance metrics calculation

### Phase 5: PPO Model
- ✅ Stable-Baselines3 PPO implementation
- ✅ GPU support with CUDA detection
- ✅ Model training with custom environment
- ✅ **NEW: Integrate hyperparam optimizer with Optuna for automated tuning**
- ✅ Training workflow: Data pipeline → Hyperparam tuning → Full training
- ✅ Model persistence: Save tuned params alongside P&L checkpoints in data/models/

### Phase 6: Dashboard
- ✅ Streamlit real-time dashboard
- ✅ 5 tabs: Live Dashboard, Top Signals, Model Insights, Performance, Jordan Chat
- ✅ Interactive charts with Plotly
- ✅ Auto-refresh every minute

### Phase 7: Jordan Personality
- ✅ Grok API integration for chat
- ✅ Self-aware personality prompts
- ✅ Daily scan and suggestions
- ✅ News monitoring and market insights

### Phase 8: Colab Integration
- ✅ GPU training notebooks
- ✅ Drive sync for models and data
- ✅ Remote training capabilities

### Phase 9: Testing Framework
- ✅ Unit tests for all components
- ✅ Integration tests
- ✅ Hyperparameter optimizer tests

### Phase 10: Deployment & Commands
- ✅ CLI scripts for training, live trading, dashboard
- ✅ Model export and import
- ✅ Production deployment ready

## 🧠 Hyperparameter Optimization

### Training Workflow
1. **Data Pipeline**: Load and preprocess historical data
2. **Hyperparameter Tuning**: Run Optuna optimization (20-200 trials)
3. **Full Training**: Use best params for long GPU simulation
4. **Model Persistence**: Save tuned model and params to data/models/

### Quick Commands
```bash
# One-time setup (installs all dependencies automatically)
python setup.py

# Local quick test
python scripts/tune_hyperparams.py 20

# Colab full optimization
# Run notebooks/hyperparam_tuning.ipynb
# Exports tuned model to Drive

# Full integration test (setup runs automatically)
python tests/run_all_tests.py

# Launch dashboard (setup runs automatically)
python scripts/simple_interface.py dashboard

# Train locally (setup runs automatically)
python scripts/simple_interface.py train_local --days 30
```

### Automatic Setup
The project now includes automatic dependency management:
- **setup.py**: Comprehensive setup script that installs all dependencies
- **Automatic execution**: All main scripts (dashboard, training, tuning, tests) run setup automatically
- **Virtual environment detection**: Warns if not using venv
- **Dependency verification**: Ensures critical packages are installed before proceeding

### Dashboard Integration
- **Tab 3 (Model Insights)**: Display current hyperparameters
- **"Tune Now" button**: Links to tuning script
- **Jordan Chat**: Daily optimization recommendations

## 📚 Documentation

- `docs/setup_guide.md`: Environment setup
- `docs/training_guide.md`: Training procedures
- `docs/hyperparam_guide.md`: Optimization guide
- `docs/user_manual.md`: User interface
- `docs/troubleshooting.md`: Common issues

## 🎯 Jordan's Take

*"Yo, these hyperparams are shit – let's optimize 'em for 20% more pips, you lazy fuck. Run the tuner weekly on fresh MT5 exports for adaptive trading. Boom – sharper than my old Stratton Oakmont suits!"*

## 💰 Let's Print Money!

Built with ❤️ for maximum profits. Use responsibly and don't blame me if you lose money – that's on you, champ! 🐺💰
