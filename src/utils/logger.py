import os
import sys
import logging
from logging.handlers import RotatingFileHandler

def get_logger(name: str = "customer_churn", log_dir: str = "logs") -> logging.Logger:
    """
    Configure and return a standardized logger that writes to both console and a rotating log file.
    """
    os.makedirs(log_dir, exist_ok=True)
    log_file_path = os.path.join(log_dir, "app.log")

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Avoid adding duplicate handlers if get_logger is called multiple times
    if not logger.handlers:
        formatter = logging.Formatter(
            "[%(asctime)s] [%(levelname)s] [%(name)s:%(lineno)d] - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )

        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        # File handler (10 MB max per file, up to 3 backups)
        file_handler = RotatingFileHandler(
            log_file_path, maxBytes=10 * 1024 * 1024, backupCount=3, encoding="utf-8"
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
