import pandas as pd

def load_historical_data(path):
    return pd.read_csv(path, parse_dates=['time'])
