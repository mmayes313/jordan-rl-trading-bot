# Delegates to technical_indicators.py
from src.indicators.technical_indicators import TechnicalIndicators

def calculate_atr(df, period):
    ti = TechnicalIndicators(df)
    return ti.compute_atr([period])
