import os
import logging
from datetime import datetime

def setup_logger(logs_dir: str = "ml/feature_engineering/logs") -> logging.Logger:
    """
    Sets up and configures the logger for the feature engineering pipeline.
    Creates a timestamped log file in the logs directory.
    """
    # Ensure logs directory exists
    os.makedirs(logs_dir, exist_ok=True)

    # Build unique log file name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(logs_dir, f"feature_engineering_pipeline_{timestamp}.log")

    # Get root logger or module logger
    logger = logging.getLogger("feature_engineering_pipeline")
    logger.setLevel(logging.DEBUG)

    # Avoid duplicate handlers if setup is called multiple times
    if logger.handlers:
        return logger

    # Create file handler
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)

    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # Create formatter
    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [%(name)s:%(lineno)d] - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    logger.info(f"Initialized feature engineering pipeline logger. Log file: {log_file}")
    
    return logger
