# logging_config.py
import logging
import sys

def setup_app_logging(name: str):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # Formato profissional com Request ID
    formatter = logging.Formatter(
        '[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)
    
    if not logger.handlers:
        logger.addHandler(handler)
    
    return logger
