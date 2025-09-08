# Delegates to technical_indicators.py
from src.indicators.technical_indicators import TechnicalIndicators

def calculate_shifted_sma(close_prices, period, shift):
    # Placeholder delegation
    return close_prices.rolling(period).mean().shift(shift)
