"""
Training Demo Script
Shows the enhanced training output with PnL, rewards, drawdown, etc.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from src.models.ppo_model import train_model_with_monitoring
from src.environment.trading_environment import TradingEnv
from src.data.data_loader import load_historical_data
import pandas as pd
import numpy as np

def create_sample_data():
    """Create sample trading data for demo."""
    # Generate synthetic OHLCV data
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=1000, freq='1H')
    
    # Generate realistic price movements
    base_price = 1.07
    returns = np.random.normal(0, 0.0001, 1000)
    prices = [base_price]
    
    for ret in returns[:-1]:
        prices.append(prices[-1] * (1 + ret))
    
    # Create OHLCV data
    data = []
    for i, (date, price) in enumerate(zip(dates, prices)):
        high = price * (1 + abs(np.random.normal(0, 0.0002)))
        low = price * (1 - abs(np.random.normal(0, 0.0002)))
        close = price + np.random.normal(0, price * 0.0001)
        volume = np.random.randint(1000, 5000)
        
        data.append({
            'timestamp': date,
            'open': price,
            'high': high,
            'low': low,
            'close': close,
            'volume': volume
        })
    
    df = pd.DataFrame(data)
    df.set_index('timestamp', inplace=True)
    return df

def demo_training():
    """Demonstrate the enhanced training with monitoring."""
    print("🎯 JORDAN RL Trading Bot - Training Demo")
    print("="*60)
    print("This demo shows the enhanced training output you'll see:")
    print("- Real-time PnL tracking")
    print("- Reward system monitoring") 
    print("- Drawdown analysis")
    print("- Daily goal achievement")
    print("- Win/loss rate calculation")
    print("- Trade count per episode")
    print("- Training health indicators")
    print("="*60)
    
    # Create sample environment
    sample_data = create_sample_data()
    env = TradingEnv(sample_data)
    
    print("\n🚀 Starting enhanced training with monitoring...")
    print("Note: This is a short demo run (1000 timesteps)")
    print("Real training would show updates every 100 timesteps\n")
    
    # Train with monitoring (short demo)
    model = train_model_with_monitoring(
        env=env,
        total_timesteps=1000,  # Short demo
        monitor_frequency=50,  # Update every 50 steps for demo
        save_path="data/models/demo_model"
    )
    
    print("\n✅ Demo completed!")
    print("\nDuring real training, you'll see:")
    print("1. 🤖 Training progress with timesteps, episodes, runtime")
    print("2. 📊 Training metrics (policy loss, value loss, entropy, KL divergence)")
    print("3. 💰 Performance metrics (avg reward, PnL, drawdown, trades/episode)")
    print("4. 🎯 Daily goals & risk management (target achievement, risk breaches)")
    print("5. 📈 Current episode status (live PnL, drawdown, trade count)")
    print("6. 🏥 Training health indicators (learning status, exploration)")
    print("\nAll metrics are also logged to data/logs/training_metrics.json")

if __name__ == "__main__":
    demo_training()
