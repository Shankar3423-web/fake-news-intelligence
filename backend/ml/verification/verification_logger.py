import os
import logging
from datetime import datetime

def setup_logger(logs_dir: str = "ml/verification/logs") -> logging.Logger:
    """
    Sets up and configures the logger for the verification pipeline.
    Creates a timestamped log file in the logs directory and configures logging.
    """
    os.makedirs(logs_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(logs_dir, f"verification_pipeline_{timestamp}.log")
    static_log_file = os.path.join(logs_dir, "verification_pipeline.log")

    logger = logging.getLogger("verification_pipeline")
    logger.setLevel(logging.DEBUG)

    # If handlers already exist, do not add more (to avoid duplicate logging)
    if logger.handlers:
        return logger

    # Timestamped file handler
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)

    # Static file handler (overwrite/append to verification_pipeline.log)
    static_file_handler = logging.FileHandler(static_log_file, mode="a", encoding="utf-8")
    static_file_handler.setLevel(logging.DEBUG)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [%(name)s:%(lineno)d] - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    file_handler.setFormatter(formatter)
    static_file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(static_file_handler)
    logger.addHandler(console_handler)

    logger.info(f"Initialized verification pipeline logger. Log file: {log_file}")
    
    return logger

def shutdown_logger() -> None:
    """
    Closes all logging handlers to release file locks (essential for Windows temp directory cleanup).
    """
    logger = logging.getLogger("verification_pipeline")
    for handler in list(logger.handlers):
        try:
            handler.close()
            logger.removeHandler(handler)
        except Exception:
            pass
