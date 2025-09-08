"""
Environment Debug Script
Diagnoses why training shows 0 episodes, 0 trades, and no learning signal.
"""

import os
import sys
# Add the project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'src'))

import numpy as np
import pandas as pd
from loguru import logger

def create_test_data():
    """Create minimal test data for debugging."""
    dates = pd.date_range('2024-01-01', periods=2000, freq='1T')
    np.random.seed(42)
    base_price = 1.07
    
    # Generate realistic price data
    prices = [base_price]
    for _ in range(1999):
        change = np.random.normal(0, 0.0001)
        new_price = prices[-1] * (1 + change)
        prices.append(new_price)
    
    df = pd.DataFrame({
        'timestamp': dates,
        'open': prices,
        'high': [p * (1 + abs(np.random.normal(0, 0.0001))) for p in prices],
        'low': [p * (1 - abs(np.random.normal(0, 0.0001))) for p in prices],
        'close': prices,
        'volume': np.random.randint(1000, 5000, 2000)
    })
    df.set_index('timestamp', inplace=True)
    return df

def test_environment():
    """Test the trading environment step by step."""
    print("🔍 ENVIRONMENT DEBUG TEST")
    print("="*50)
    
    # Test data loading and processing
    try:
        from data.data_loader import load_historical_data
        print("✅ Attempting to load real data...")
        df = load_historical_data('data/raw/EURUSD_1.csv')
        print(f"✅ Loaded real data: {df.shape}")
    except Exception as e:
        print(f"⚠️ Real data failed ({e}), using synthetic data")
        df = create_test_data()
        print(f"✅ Created synthetic data: {df.shape}")
    
    # Test data processing
    print("\n📊 Testing Data Processor...")
    try:
        from src.data.data_processor import DataProcessor
        processor = DataProcessor(df)
        processed_data = processor.preprocess()
        print(f"✅ Data processed: {processed_data.shape}")
        print(f"   NaNs: {processed_data.isna().sum().sum()}")
        print(f"   Data range: {processed_data.min().min():.6f} to {processed_data.max().max():.6f}")
    except Exception as e:
        print(f"❌ Data processing failed: {e}")
        return
    
    # Test technical indicators
    print("\n📈 Testing Technical Indicators...")
    try:
        from src.indicators.technical_indicators import TechnicalIndicators
        indicators = TechnicalIndicators(df)
        cci_features = indicators.compute_cci(periods=[30])
        print(f"✅ CCI computed: {cci_features.shape}")
        print(f"   CCI range: {cci_features.min().min():.2f} to {cci_features.max().max():.2f}")
    except Exception as e:
        print(f"❌ Indicators failed: {e}")
        cci_features = pd.DataFrame({'CCI_30': np.random.normal(0, 50, len(df))})
    
    # Test trading masks
    print("\n🎭 Testing Trading Masks...")
    try:
        from src.masks.trading_masks import TradingMasks
        cci_values = cci_features['CCI_30'].values
        masks = TradingMasks(cci_values)
        
        # Test mask with different slices
        test_results = []
        for i in range(0, min(100, len(cci_values)-5), 10):
            slice_result = masks.apply_mask(cci_values[i:i+3])
            test_results.append(slice_result)
        
        allowed_count = sum(test_results)
        print(f"✅ Masks tested: {allowed_count}/{len(test_results)} trades allowed")
        print(f"   Training mode: {getattr(masks, 'training_mode', 'NOT_SET')}")
        
        if allowed_count == 0:
            print("⚠️ WARNING: All trades blocked by masks!")
            print("   Setting training mode...")
            masks.training_mode = True
            test_results_training = []
            for i in range(0, min(100, len(cci_values)-5), 10):
                slice_result = masks.apply_mask(cci_values[i:i+3])
                test_results_training.append(slice_result)
            allowed_training = sum(test_results_training)
            print(f"   With training mode: {allowed_training}/{len(test_results_training)} trades allowed")
            
    except Exception as e:
        print(f"❌ Masks failed: {e}")
    
    # Test environment creation and basic operations
    print("\n🏗️ Testing Trading Environment...")
    try:
        from src.environment.complete_trading_env import CompleteTradingEnv
        env = CompleteTradingEnv(data=None, max_episode_steps=500)
        print(f"✅ Environment created")
        print(f"   Observation space: {env.observation_space}")
        print(f"   Action space: {env.action_space}")
        
        # Test reset
        obs, info = env.reset()
        print(f"✅ Environment reset")
        print(f"   Obs shape: {obs.shape if hasattr(obs, 'shape') else type(obs)}")
        print(f"   Obs range: {np.min(obs):.6f} to {np.max(obs):.6f}")
        print(f"   NaNs in obs: {np.isnan(obs).sum()}")
        
    except Exception as e:
        print(f"❌ Environment creation failed: {e}")
        return
    
    # Test environment steps
    print("\n🚶 Testing Environment Steps...")
    episode_rewards = []
    episode_length = 0
    total_reward = 0
    trade_count = 0
    done_occurred = False
    
    for step in range(500):  # Test 500 steps max
        try:
            action = env.action_space.sample()
            result = env.step(action)
            
            # Handle both 4-tuple and 5-tuple returns
            if len(result) == 4:
                next_obs, reward, done, info = result
                truncated = False
            else:
                next_obs, reward, done, truncated, info = result
            
            total_reward += reward
            episode_length += 1
            
            # Check for trades in info
            if 'trade_outcome' in info or 'pnl' in info:
                trade_count += 1
            
            # Log every 50 steps
            if step % 50 == 0:
                print(f"   Step {step}: action={action}, reward={reward:.6f}, done={done}")
                print(f"     Obs range: {np.min(next_obs):.6f} to {np.max(next_obs):.6f}")
                if info:
                    print(f"     Info: {info}")
            
            # Check for NaNs
            if np.isnan(next_obs).any():
                print(f"❌ NaN detected in observations at step {step}")
                break
            
            # Check for episode end
            if done or truncated:
                print(f"✅ Episode ended at step {step} (done={done}, truncated={truncated})")
                done_occurred = True
                break
                
        except Exception as e:
            print(f"❌ Step {step} failed: {e}")
            break
    
    # Summary
    print(f"\n📋 ENVIRONMENT TEST SUMMARY:")
    print(f"   Episode length: {episode_length}")
    print(f"   Total reward: {total_reward:.6f}")
    print(f"   Trade count: {trade_count}")
    print(f"   Episode ended naturally: {done_occurred}")
    print(f"   Avg reward per step: {total_reward/max(1, episode_length):.6f}")
    
    # Diagnose issues
    print(f"\n🩺 DIAGNOSIS:")
    issues = []
    
    if not done_occurred:
        issues.append("❌ Episode never ends (no done=True)")
    if total_reward == 0:
        issues.append("❌ No rewards generated")
    if trade_count == 0:
        issues.append("❌ No trades executed")
    if episode_length == 0:
        issues.append("❌ Environment not stepping")
    
    if not issues:
        print("✅ Environment appears to be working correctly!")
    else:
        print("⚠️ Issues found:")
        for issue in issues:
            print(f"   {issue}")
    
    return len(issues) == 0

if __name__ == "__main__":
    success = test_environment()
    if success:
        print("\n🎉 Environment debug completed successfully!")
    else:
        print("\n🔧 Environment needs fixes - see diagnosis above.")
