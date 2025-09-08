import sys
from loguru import logger

# Configure logger with file rotation
logger.add('../../data/logs/bot.log', level='DEBUG', rotation='1 day', retention='7 days')
logger.add(sys.stdout, level='INFO')

def get_logger():
    """Get the configured logger instance"""
    return logger
