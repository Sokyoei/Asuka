import sys
from pathlib import Path

from loguru import logger

FORMATTER = (
    "[<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green>][<cyan>{file.path}:{line}</cyan>]"
    "[<level>{level}</level>]: <level>{message}</level>"
)


def init_logging(log_path: Path | None = None):
    from Ahri.Asuka.config.config import settings

    logger.remove(handler_id=None)
    logger.add(sys.stderr, format=FORMATTER, colorize=True)
    logger.add(
        log_path if log_path else settings.LOG_DIR / "Ahri.Asuka_{time:YYYY-MM-DD}.log",
        format=FORMATTER,
        rotation="00:00",
        retention="30 days",
        # compression="zip",
        enqueue=True,
    )
