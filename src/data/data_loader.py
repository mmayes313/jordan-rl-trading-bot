import pandas as pd
import numpy as np

def load_historical_data(filepath):
    """
    Load historical OHLCV data from CSV.
    Expected columns: timestamp, open, high, low, close, volume
    """
    df = pd.read_csv(filepath)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.set_index('timestamp', inplace=True)
    return df

if __name__ == "__main__":
    # Example usage
    data = load_historical_data('data/raw/eurusd_1y.csv')
    print(data.head())
