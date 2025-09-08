"""
Technical Indicators Module
Combines all technical indicator calculations for the trading bot.
"""

import pandas as pd
import numpy as np
from loguru import logger

# Import individual indicator modules
from .cci_indicators import calculate_cci, smooth_cci
from .sma_indicators import calculate_shifted_sma
from .atr_indicators import calculate_atr, smooth_atr
from .adx_obv_indicators import calculate_adx, calculate_obv

class TechnicalIndicators:
    """
    Comprehensive technical indicators calculator.
    Combines CCI, SMA, ATR, ADX, and OBV indicators across multiple timeframes.
    """

    def __init__(self, df):
        """
        Initialize with OHLCV data.

        Args:
            df: DataFrame with columns ['open', 'high', 'low', 'close', 'volume']
        """
        self.df = df.copy()
        self.validate_data()
        logger.info(f"TechnicalIndicators initialized with {len(df)} data points")

    def validate_data(self):
        """Validate that required columns exist."""
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        missing_cols = [col for col in required_cols if col not in self.df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

    def compute_cci(self, periods=[3, 5, 30, 100, 300]):
        """
        Compute CCI indicators for multiple periods.

        Args:
            periods: List of CCI periods to compute

        Returns:
            DataFrame with CCI features
        """
        cci_features = []
        for period in periods:
            cci = calculate_cci(self.df, period)
            cci_features.append(cci.rename(f'CCI_{period}'))

            # Add smoothed versions
            for sma_period in [2, 5]:
                smoothed = smooth_cci(cci, sma_period=sma_period)
                cci_features.append(smoothed.rename(f'CCI_{period}_SMA{sma_period}'))

        result = pd.concat(cci_features, axis=1)
        logger.info(f"Computed {len(cci_features)} CCI features")
        return result

    def compute_sma(self, periods=[1, 2, 3, 5, 10, 20, 50, 200], shifts=None):
        """
        Compute SMA indicators with various shifts.

        Args:
            periods: List of SMA periods
            shifts: List of shift values (positive=forward, negative=backward)

        Returns:
            DataFrame with SMA features
        """
        if shifts is None:
            # Default shifts: current + forward/backward shifts
            shifts = [0, 1, 2, 3, 4, 5, -1, -2, -3, -4, -5]

        sma_features = []
        data_length = len(self.df)

        for period in periods:
            # Skip periods that are longer than available data
            if period >= data_length:
                logger.warning(f"Skipping SMA period {period} - data length {data_length} too short")
                continue

            for shift in shifts:
                try:
                    sma = calculate_shifted_sma(self.df['close'], period, shift)
                    shift_name = f"F{shift}" if shift > 0 else f"B{-shift}" if shift < 0 else ""
                    name = f'SMA_{period}{shift_name}'
                    sma_features.append(sma.rename(name))
                except Exception as e:
                    logger.warning(f"Failed to compute SMA_{period} with shift {shift}: {e}")
                    continue

        if not sma_features:
            # If no SMAs could be computed, return empty DataFrame with proper structure
            return pd.DataFrame(index=self.df.index)

        result = pd.concat(sma_features, axis=1)
        logger.info(f"Computed {len(sma_features)} SMA features")
        return result

    def compute_atr(self, periods=[5, 14]):
        """
        Compute ATR indicators for multiple periods.

        Args:
            periods: List of ATR periods

        Returns:
            DataFrame with ATR features
        """
        atr_features = []
        data_length = len(self.df)

        for period in periods:
            # Skip periods that are longer than available data
            if period >= data_length:
                logger.warning(f"Skipping ATR period {period} - data length {data_length} too short")
                continue

            try:
                atr = calculate_atr(self.df, period)
                atr_features.append(atr.rename(f'ATR_{period}'))

                # Add smoothed versions
                smoothed = smooth_atr(atr, sma_period=2)
                atr_features.append(smoothed.rename(f'ATR_{period}_SMA2'))
            except Exception as e:
                logger.warning(f"Failed to compute ATR_{period}: {e}")
                continue

        if not atr_features:
            # If no ATRs could be computed, return empty DataFrame
            return pd.DataFrame(index=self.df.index)

        result = pd.concat(atr_features, axis=1)
        logger.info(f"Computed {len(atr_features)} ATR features")
        return result

    def compute_adx(self, period=14):
        """
        Compute ADX indicator.

        Args:
            period: ADX period

        Returns:
            DataFrame with ADX features
        """
        adx = calculate_adx(self.df, period)
        result = pd.DataFrame({
            f'ADX_{period}': adx
        })
        logger.info(f"Computed ADX_{period} feature")
        return result

    def compute_obv(self):
        """
        Compute OBV (On Balance Volume) indicator.

        Returns:
            DataFrame with OBV features
        """
        obv = calculate_obv(self.df)
        result = pd.DataFrame({
            'OBV': obv
        })
        logger.info("Computed OBV feature")
        return result

    def compute_all_indicators(self):
        """
        Compute all technical indicators.

        Returns:
            DataFrame with all indicator features
        """
        logger.info("Computing all technical indicators...")

        # Compute each indicator type
        cci_features = self.compute_cci()
        sma_features = self.compute_sma()
        atr_features = self.compute_atr()
        adx_features = self.compute_adx()
        obv_features = self.compute_obv()

        # Combine all features
        all_features = pd.concat([
            cci_features,
            sma_features,
            atr_features,
            adx_features,
            obv_features
        ], axis=1)

        logger.info(f"Total features computed: {all_features.shape[1]}")
        return all_features
