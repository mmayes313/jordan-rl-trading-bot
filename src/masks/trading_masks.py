import pandas as pd

class TradingMasks:
    def __init__(self, indicators):
        self.indicators = indicators

    def apply_mask(self, cci_periods=[30,100], training_mode=False):
        if training_mode: return True
        
        # Mock implementation for CCI values (in real implementation, would use proper timeframe data)
        cci_30_1m = 50  # Mock value
        cci_100_1m = 50  # Mock value
        cci_30_15m = 50  # Mock value
        cci_100_15m = 50  # Mock value
        cci_30_1m_sma2 = 50  # Mock value
        cci_100_1m_sma2 = 50  # Mock value
        cci_30_15m_sma2 = 50  # Mock value
        cci_100_15m_sma2 = 50  # Mock value
        
        cci_values = [cci_30_1m, cci_100_1m, cci_30_15m, cci_100_15m, cci_30_1m_sma2, cci_100_1m_sma2, cci_30_15m_sma2, cci_100_15m_sma2]

        sell_block = all(v > 60 for v in cci_values)  # All 8 true to block SELL
        buy_block = all(v < -60 for v in cci_values)  # All 8 true to block BUY
        return not (buy_block or sell_block)  # Allow if not blocked
