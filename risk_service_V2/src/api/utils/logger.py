from loguru import logger


def setup_logging():
    logger.remove()
    logger.add(lambda msg: print(msg, end=""), level="INFO")
    logger.info("API logger initialized")