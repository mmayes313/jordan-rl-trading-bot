import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import os

def normalize_time(hour):
    """Normalize hour to [0,1] range"""
    return hour / 24.0

def normalize_price(price, base_price=1.0):
    """Normalize price relative to base price"""
    return price / base_price - 1.0

def calculate_returns(prices):
    """Calculate percentage returns from price series"""
    return np.diff(prices) / prices[:-1]

def calculate_volatility(returns, window=20):
    """Calculate rolling volatility"""
    return np.std(returns[-window:]) * np.sqrt(252)  # Annualized

def calculate_sharpe_ratio(returns, risk_free_rate=0.02):
    """Calculate Sharpe ratio"""
    excess_returns = returns - risk_free_rate / 252
    if np.std(excess_returns) == 0:
        return 0
    return np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252)

def format_currency(amount):
    """Format amount as currency string"""
    return f"${amount:,.2f}"

def format_percentage(value):
    """Format value as percentage string"""
    return f"{value:.2f}%"

def safe_divide(a, b, default=0):
    """Safe division that returns default if denominator is zero"""
    return a / b if b != 0 else default

def clamp(value, min_val, max_val):
    """Clamp value between min and max"""
    return max(min_val, min(value, max_val))

def moving_average(data, window):
    """Calculate simple moving average"""
    if len(data) < window:
        return np.mean(data)
    return np.mean(data[-window:])

def exponential_moving_average(data, window):
    """Calculate exponential moving average"""
    if len(data) < window:
        return np.mean(data)
    alpha = 2 / (window + 1)
    ema = np.zeros(len(data))
    ema[0] = data[0]
    for i in range(1, len(data)):
        ema[i] = alpha * data[i] + (1 - alpha) * ema[i-1]
    return ema[-1]

def create_directory_if_not_exists(path):
    """Create directory if it doesn't exist"""
    os.makedirs(path, exist_ok=True)
    return path

def get_project_root():
    """Get the project root directory"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.dirname(os.path.dirname(current_dir))

def load_config_from_env():
    """Load configuration from environment variables"""
    config = {}
    # Add environment variable loading logic here
    return config

def validate_dataframe(df, required_columns):
    """Validate that dataframe has required columns"""
    missing_columns = set(required_columns) - set(df.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    return True

def resample_timeframe(data, timeframe):
    """Resample data to different timeframe"""
    # Implementation for resampling OHLCV data
    if timeframe == 'H':
        return data.resample('H').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        })
    return data
