from loguru import logger
import os

log_dir = os.path.join(os.path.dirname(__file__), '../../data/logs')
os.makedirs(log_dir, exist_ok=True)

logger.add(os.path.join(log_dir, 'trading.log'), rotation="1 day")
