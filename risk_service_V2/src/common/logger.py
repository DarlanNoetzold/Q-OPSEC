from pathlib import Path
from typing import Optional

from loguru import logger


_LOGGER_CONFIGURED = False


def setup_logger(log_dir: str | Path = "logs", level: str = "INFO") -> None:
    """Configure loguru logger with console + rotating file.

    Should be called once at startup.
    """
    global _LOGGER_CONFIGURED
    if _LOGGER_CONFIGURED:
        return

    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "dataset_generation.log"

    logger.remove()  # remove default handler

    # Console handler
    logger.add(
        sink=lambda msg: print(msg, end=""),
        level=level.upper(),
        colorize=True,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
        "<level>{message}</level>",
    )

    # File handler (rotating)
    logger.add(
        log_file,
        level=level.upper(),
        rotation="10 MB",
        retention="10 days",
        compression="zip",
        enqueue=True,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
    )

    _LOGGER_CONFIGURED = True


def get_logger(name: Optional[str] = None):
    """Return loguru logger (optionally bind a name)."""
    if name:
        return logger.bind(component=name)
    return logger
