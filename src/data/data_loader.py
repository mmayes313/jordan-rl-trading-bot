import pandas as pd

def load_historical_data(path):
    data = pd.read_csv(path, parse_dates=['time'])
    if 'time' in data.columns:
        data = data.set_index('time')
    return data
