#!/usr/bin/env python
# debug_imports.py - Debug import issues for testing

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

print("Python path:")
for p in sys.path:
    print(f"  {p}")

print("\nTesting imports...")

try:
    from indicators.technical_indicators import TechnicalIndicators
    print("✅ TechnicalIndicators imported successfully")
except ImportError as e:
    print(f"❌ TechnicalIndicators import failed: {e}")

try:
    from environment.trading_environment import TradingEnv
    print("✅ TradingEnv imported successfully")
except ImportError as e:
    print(f"❌ TradingEnv import failed: {e}")

try:
    from rewards.reward_system import RewardSystem
    print("✅ RewardSystem imported successfully")
except ImportError as e:
    print(f"❌ RewardSystem import failed: {e}")

try:
    from masks.trading_masks import TradingMasks
    print("✅ TradingMasks imported successfully")
except ImportError as e:
    print(f"❌ TradingMasks import failed: {e}")

try:
    from models.ppo_model import create_ppo_model
    print("✅ create_ppo_model imported successfully")
except ImportError as e:
    print(f"❌ create_ppo_model import failed: {e}")

print("\nImport debugging complete.")
