from loguru import logger


def setup_logging():
    # Basic configuration - integrate with existing logger if needed
    logger.remove()
    logger.add(lambda msg: print(msg, end=""), level="INFO")
    logger.info("API logger initialized")