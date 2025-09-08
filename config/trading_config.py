import os
from dotenv import load_dotenv
load_dotenv()

DAILY_TARGET_PCT = float(os.getenv('DAILY_TARGET_PCT', 0.10))  # Default 10%, override via .env or dashboard
TRAILING_DD_PCT = float(os.getenv('TRAILING_DD_PCT', 0.05))    # Default 5%
SPREAD_MULTIPLIER = 1.5
TIMEFRAMES = [1, 15, 60, 1440]  # 1m,15m,1h,1d
GROK_API_KEY = os.getenv('GROK_API_KEY')  # Your key here
SYMBOLS_FILE = 'data/raw/all_symbols.pkl'  # Cache if needed

ASSETS = [
    # List all MT5 symbols you plan to trade, e.g.:
    'EURUSD', 'USDJPY', 'GBPUSD', 'AUDUSD', 'USDCAD', 'USDCHF', 'NZDUSD',
    'XAUUSD', 'XAGUSD', 'WTICOUSD', 'US30', 'NAS100', 'SPX500',
    # Add more as needed
]