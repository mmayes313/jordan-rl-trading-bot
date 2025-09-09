#!/usr/bin/env python3
"""Test script for trading environment"""

import warnings
warnings.filterwarnings("ignore")

try:
    from src.environment.trading_environment import TradingEnvironment
    print("✓ Successfully imported TradingEnvironment")
except Exception as e:
    print(f"✗ Import error: {e}")
    exit(1)

def test_environment():
    try:
        print("Creating trading environment...")
        env = TradingEnvironment('data/raw/eurusd_sample.csv')
        print("✓ Environment created successfully")
        
        print("Resetting environment...")
        obs, info = env.reset()
        print("✓ Environment reset successfully")
        
        print(f"Observation shape: {obs.shape}")
        print(f"Expected shape: (417,)")
        print(f"Shape matches: {obs.shape == (417,)}")
        
        if obs.shape == (417,):
            print("✓ Observation space is correct!")
        else:
            print("✗ Observation space mismatch")
        
        print("\nTesting step...")
        action = [0.1]  # Test lot size
        obs2, reward, terminated, truncated, info = env.step(action)
        print(f"✓ Step completed successfully")
        print(f"Step 2 observation shape: {obs2.shape}")
        print(f"Reward: {reward}")
        print(f"Info: {info}")
        
        print("\n🎉 Environment test completed successfully!")
        
    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_environment()
