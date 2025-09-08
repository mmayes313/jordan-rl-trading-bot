import MetaTrader5 as mt5
from src.environment.mt5_connector import connect_mt5, get_rates

if connect_mt5():
    rates = get_rates('EURUSD', mt5.TIMEFRAME_M1, 525600)  # 1y 1m
    rates.to_csv('data/raw/eurusd_1y.csv')
    print("Data exported!")
