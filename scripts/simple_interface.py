import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import argparse
import subprocess
from pathlib import Path

def run_setup():
    """Run the setup script to ensure dependencies are installed."""
    setup_path = Path(__file__).parent.parent / 'setup.py'
    print("🔧 Ensuring dependencies are installed...")
    try:
        result = subprocess.run([sys.executable, str(setup_path)],
                               check=True, capture_output=False, timeout=600)  # 10 minute timeout
        return True
    except subprocess.TimeoutExpired:
        print("⏰ Setup timed out after 10 minutes")
        return False
    except subprocess.CalledProcessError as e:
        print(f"❌ Setup failed: {e}")
        return False

# Run setup first
if not run_setup():
    print("❌ Cannot continue without proper setup")
    sys.exit(1)

import argparse
import subprocess
from src.environment.trading_environment import TradingEnv
from src.models.ppo_model import train_model, train_model_with_monitoring, PPO
from src.mt5_connector import export_ohlcv
from config.trading_config import ASSETS, TIMEFRAMES

parser = argparse.ArgumentParser()
subparsers = parser.add_subparsers()

# Train local
train_local = subparsers.add_parser('train_local')
train_local.add_argument('--days', type=int, default=100)
train_local.add_argument('--monitor-freq', type=int, default=100, help='Display update frequency (timesteps)')
train_local.set_defaults(func=lambda args: train_model_with_monitoring(
    TradingEnv({}), 
    total_timesteps=args.days*1440*60, 
    monitor_frequency=args.monitor_freq
))

# Train local (legacy - no monitoring)
train_simple = subparsers.add_parser('train_simple')
train_simple.add_argument('--days', type=int, default=100)
train_simple.set_defaults(func=lambda args: train_model(TradingEnv({}), total_timesteps=args.days*1440*60))

# Train Colab (run in notebook)
# Live trade
live = subparsers.add_parser('live')
live.set_defaults(func=lambda args: TradingEnv({}).run_live(model=PPO.load('data/models/best_model')))

# Dashboard
dash = subparsers.add_parser('dashboard')
dash.set_defaults(func=lambda args: subprocess.call(['streamlit', 'run', 'streamlit/app.py']))

# Export data
export = subparsers.add_parser('export')
export.set_defaults(func=lambda args: [export_ohlcv(s, tf) for s in ASSETS for tf in TIMEFRAMES])

# Daily scan
daily = subparsers.add_parser('daily')
daily.set_defaults(func=lambda args: daily_func())

def daily_func():
    from streamlit.components.jordan_personality import daily_scan_and_suggest, monitor_news
    return daily_scan_and_suggest(), monitor_news()

args = parser.parse_args()
if hasattr(args, 'func'): args.func(args)
