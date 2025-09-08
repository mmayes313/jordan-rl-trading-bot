import MetaTrader5 as mt5
import pandas as pd
from loguru import logger
import talib

def connect_mt5():
    if not mt5.initialize():
        logger.error("MT5 init failed")
        return False
    logger.info("MT5 connected – time to make money!")
    return True

def export_ohlcv(symbol, timeframe, count=10000):
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, count)
    df = pd.DataFrame(rates)
    df.to_csv(f"data/raw/{symbol}_{timeframe}.csv")
    return df

def place_trade(symbol, action, lots=0.01):  # Bot controls lots
    if action == "buy":
        request = { "action": mt5.TRADE_ACTION_DEAL, "symbol": symbol, "volume": lots, "type": mt5.ORDER_TYPE_BUY }
    # Similar for sell
    result = mt5.order_send(request)
    logger.info(f"Traded {lots} lots {action} on {symbol}")
    return result

def get_all_symbols():
    """Dynamically fetch all tradable symbols from current broker."""
    if not mt5.initialize():
        logger.error("MT5 init failed")
        return []
    symbols = mt5.symbols_get()
    tradable = [s.name for s in symbols if s.visible and s.trade_mode == mt5.SYMBOL_TRADE_MODE_FULL]  # Only tradable
    logger.info(f"Fetched {len(tradable)} symbols: {tradable[:10]}...")  # Log first 10
    mt5.shutdown()
    return tradable

def rank_fast_movers(symbols, tf=1):  # 1m for momentum
    """Rank by ATR (volatility/fast moves) + low spread. For top signals/selection."""
    mt5.initialize()
    ranked = []
    for sym in symbols[:50]:  # Top 50 to avoid overload
        rates = mt5.copy_rates_from_pos(sym, tf, 0, 100)  # Recent data
        if len(rates) < 14: continue
        df = pd.DataFrame(rates)
        atr = talib.ATR(df['high'], df['low'], df['close'], timeperiod=14)[-1]  # Volatility
        symbol_info = mt5.symbol_info(sym)
        spread = symbol_info.spread * symbol_info.point  # In price units
        score = atr / (spread + 1e-6)  # High vol / low spread = fast profit potential
        ranked.append({'symbol': sym, 'score': score, 'atr': atr, 'spread': spread})
    mt5.shutdown()
    return sorted(ranked, key=lambda x: x['score'], reverse=True)  # Top fast movers first
