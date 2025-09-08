from pydantic import BaseModel

class TradingConfig(BaseModel):
    mt5_login: int = 123456  # Your MT5 account ID
    mt5_password: str = "yourpass"  # Redact for Git
    server: str = "YourBroker-Demo"
    initial_balance: float = 10000.0
    max_lot_size: float = 10.0
    min_lot_size: float = 0.01
    daily_target: float = 0.01  # 1% profit
    max_drawdown: float = 0.05  # 5%