import logging
from pathlib import Path


def setup_logger(name: str, log_level: str = "INFO") -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(log_level.upper())
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    logs_dir = Path(__file__).resolve().parents[1] / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(logs_dir / "app.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    logger.propagate = False
    return logger
