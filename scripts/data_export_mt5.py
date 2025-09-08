import argparse
from src.mt5_connector import connect_mt5, export_ohlcv
from config.trading_config import ASSETS, TIMEFRAMES

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--symbols', type=str, help='Comma-separated list of symbols', default=','.join(ASSETS))
    parser.add_argument('--timeframe', type=int, help='MT5 timeframe (e.g., 1, 15, 60, 1440)', default=1)
    args = parser.parse_args()

    connect_mt5()
    for symbol in args.symbols.split(','):
        export_ohlcv(symbol.strip(), args.timeframe)
