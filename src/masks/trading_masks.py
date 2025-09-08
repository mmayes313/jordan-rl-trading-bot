"""
Trading Masks Module
Implements CCI-based trading masks to prevent trades during extreme market conditions.
"""

import numpy as np
from loguru import logger

class TradingMasks:
    """
    CCI-based trading masks to block trades during extreme overbought/oversold conditions.
    Prevents trading when CCI indicates unsustainable market extremes.
    """

    def __init__(self, cci_values=None, training_mode=False):
        """
        Initialize trading masks with CCI values.

        Args:
            cci_values: Array of CCI values for mask calculation
            training_mode: If True, relax thresholds for training
        """
        self.cci_values = cci_values
        self.training_mode = training_mode

        # CCI thresholds for blocking trades
        if self.training_mode:
            # Relaxed thresholds during training to allow more trades
            self.buy_block_threshold = 100  # Block buy when CCI > 100
            self.sell_block_threshold = -100  # Block sell when CCI < -100
        else:
            # Stricter thresholds for live trading
            self.buy_block_threshold = 60  # Block buy when CCI > 60
            self.sell_block_threshold = -60  # Block sell when CCI < -60

        logger.info(f"TradingMasks initialized with training_mode={training_mode}")
        logger.info(f"Buy block threshold: CCI > {self.buy_block_threshold}")
        logger.info(f"Sell block threshold: CCI < {self.sell_block_threshold}")

    def apply_mask(self, cci_window):
        """
        Apply trading mask based on CCI conditions.

        Args:
            cci_window: Array of recent CCI values (typically 8 values)

        Returns:
            bool: True if trading is allowed, False if blocked
        """
        if len(cci_window) < 8:
            logger.warning(f"CCI window too short: {len(cci_window)} values, need 8")
            return False

        # Extract last 8 CCI values for mask logic
        recent_cci = cci_window[-8:]

        # Check buy mask conditions (block when overbought)
        buy_blocked = all(cci > self.buy_block_threshold for cci in recent_cci)

        # Check sell mask conditions (block when oversold)
        sell_blocked = all(cci < self.sell_block_threshold for cci in recent_cci)

        # Trade is allowed if neither buy nor sell is blocked
        trade_allowed = not (buy_blocked or sell_blocked)

        logger.debug(f"Mask check: CCI={recent_cci}, Buy_blocked={buy_blocked}, Sell_blocked={sell_blocked}, Allowed={trade_allowed}")

        return trade_allowed

    def get_mask_stats(self, cci_values):
        """
        Get statistics about mask application over a series of CCI values.

        Args:
            cci_values: Array of CCI values

        Returns:
            dict: Statistics about mask blocking
        """
        total_checks = len(cci_values) - 7  # Need 8 values for each check
        blocked_trades = 0
        allowed_trades = 0

        for i in range(total_checks):
            window = cci_values[i:i+8]
            if self.apply_mask(window):
                allowed_trades += 1
            else:
                blocked_trades += 1

        return {
            'total_checks': total_checks,
            'allowed_trades': allowed_trades,
            'blocked_trades': blocked_trades,
            'block_percentage': (blocked_trades / total_checks * 100) if total_checks > 0 else 0
        }
