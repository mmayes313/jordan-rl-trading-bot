import pandas as pd
try:
    import MetaTrader5 as mt5
except ImportError:
    print("MetaTrader5 not installed. Using mock functions.")
    mt5 = None

from config.trading_config import TradingConfig
config = TradingConfig()

def connect_mt5():
    if mt5 is None:
        print("MT5 module not available, using mock connection")
        return True
    
    if not mt5.initialize(login=config.mt5_login, password=config.mt5_password, server=config.server):
        print("MT5 init failed")
        return False
    return True

def get_rates(symbol, timeframe, count):
    if mt5 is None:
        # Mock data
        import numpy as np
        mock_data = {
            'time': range(count),
            'open': np.random.uniform(1.1, 1.11, count),
            'high': np.random.uniform(1.11, 1.12, count),
            'low': np.random.uniform(1.09, 1.1, count),
            'close': np.random.uniform(1.1, 1.11, count),
            'tick_volume': np.random.randint(100, 1000, count)
        }
        return pd.DataFrame(mock_data)
    
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, count)
    return pd.DataFrame(rates)
