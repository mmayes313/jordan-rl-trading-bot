"""
Data Processor Module
Handles data preprocessing and normalization for the trading bot.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from loguru import logger
import joblib

def preprocess(data):
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaled = scaler.fit_transform(data.select_dtypes(include=[np.number]))
    joblib.dump(scaler, '../data/models/scaler.pkl')
    return scaled

class DataProcessor:
    """
    Data preprocessing and normalization for trading data.
    Handles OHLCV data and prepares it for model training.
    """

    def __init__(self, data):
        """
        Initialize with raw OHLCV data.

        Args:
            data: DataFrame with OHLCV columns
        """
        self.raw_data = data.copy()
        self.processed_data = None
        self.scalers = {}
        logger.info(f"DataProcessor initialized with {len(data)} data points")

    def validate_data(self):
        """Validate that required columns exist."""
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        missing_cols = [col for col in required_cols if col not in self.raw_data.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        # Check for NaN values
        nan_count = self.raw_data.isna().sum().sum()
        if nan_count > 0:
            logger.warning(f"Found {nan_count} NaN values in data")

    def preprocess(self, method='standard'):
        """
        Preprocess and normalize the data.

        Args:
            method: Normalization method ('standard', 'minmax', or 'robust')

        Returns:
            DataFrame: Processed and normalized data
        """
        logger.info(f"Preprocessing data with method: {method}")
        self.validate_data()

        # Start with raw data
        processed = self.raw_data.copy()

        # Handle missing values
        processed = processed.fillna(method='ffill').fillna(method='bfill')

        # Normalize price data
        price_cols = ['open', 'high', 'low', 'close']
        if method == 'standard':
            scaler = StandardScaler()
        elif method == 'minmax':
            scaler = MinMaxScaler(feature_range=(-1, 1))
        else:
            # Default to standard scaling
            scaler = StandardScaler()

        # Fit and transform price columns
        processed[price_cols] = scaler.fit_transform(processed[price_cols])
        self.scalers['prices'] = scaler

        # Normalize volume (often needs different scaling)
        if 'volume' in processed.columns:
            vol_scaler = StandardScaler()
            processed['volume'] = vol_scaler.fit_transform(processed[['volume']])
            self.scalers['volume'] = vol_scaler

        # Add derived features
        processed = self.add_derived_features(processed)

        self.processed_data = processed
        logger.info(f"Data preprocessing complete. Shape: {processed.shape}")
        return processed

    def add_derived_features(self, df):
        """
        Add derived features like returns, volatility measures, etc.

        Args:
            df: DataFrame with OHLCV data

        Returns:
            DataFrame: DataFrame with additional features
        """
        # Price returns
        df['returns'] = df['close'].pct_change()

        # High-low range
        df['range'] = (df['high'] - df['low']) / df['close']

        # Body size
        df['body_size'] = abs(df['close'] - df['open']) / df['close']

        # Upper/lower wick ratios
        df['upper_wick'] = (df['high'] - df[['open', 'close']].max(axis=1)) / df['close']
        df['lower_wick'] = (df[['open', 'close']].min(axis=1) - df['low']) / df['close']

        # Fill any NaN values from pct_change
        df = df.fillna(0)

        return df

    def inverse_transform_prices(self, normalized_prices):
        """
        Inverse transform normalized prices back to original scale.

        Args:
            normalized_prices: Normalized price values

        Returns:
            Array: Original scale prices
        """
        if 'prices' not in self.scalers:
            raise ValueError("No price scaler found. Run preprocess() first.")

        return self.scalers['prices'].inverse_transform(normalized_prices)

    def get_data_stats(self):
        """
        Get statistics about the processed data.

        Returns:
            dict: Data statistics
        """
        if self.processed_data is None:
            raise ValueError("No processed data found. Run preprocess() first.")

        stats = {
            'shape': self.processed_data.shape,
            'columns': list(self.processed_data.columns),
            'nan_count': self.processed_data.isna().sum().sum(),
            'price_mean': self.processed_data[['open', 'high', 'low', 'close']].mean().mean(),
            'price_std': self.processed_data[['open', 'high', 'low', 'close']].std().mean(),
            'volume_mean': self.processed_data['volume'].mean() if 'volume' in self.processed_data.columns else None,
            'volume_std': self.processed_data['volume'].std() if 'volume' in self.processed_data.columns else None,
        }

        return stats
