from src.mt5_connector import get_all_symbols, rank_fast_movers, export_ohlcv
from src.indicators.cci_indicators import calculate_cci

def get_top_signals():
    symbols = get_all_symbols()
    fast_ranked = rank_fast_movers(symbols)
    top10 = fast_ranked[:10]
    signals = []
    for item in top10:
        # Compute momentum: 5m/30m CCI/ADX score
        df5m = export_ohlcv(item['symbol'], 5)  # Quick fetch
        cci5m = calculate_cci(df5m, 30)[-1]
        # Similar for 30m, score = abs(cci) if >60 else 0 + ADX
        confidence = min(100, abs(cci5m) * 0.5)  # Simple
        exp_pnl = item['atr'] * 2  # Rough risk/reward
        signals.append({'asset': item['symbol'], 'direction': 'buy' if cci5m > 0 else 'sell', 'score': confidence, 'pnl': exp_pnl, 'reason': f"Fast mover: High ATR {item['atr']:.4f}, tight spread {item['spread']:.5f}"})
    return signals
