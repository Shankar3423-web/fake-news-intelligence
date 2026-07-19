import os
import logging
from datetime import datetime

def setup_logger(logs_dir: str = "ml/evaluation/logs") -> logging.Logger:
    """
    Sets up and configures the logger for the model evaluation pipeline.
    Creates a timestamped log file in the logs directory.
    """
    os.makedirs(logs_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(logs_dir, f"model_evaluation_pipeline_{timestamp}.log")

    logger = logging.getLogger("model_evaluation_pipeline")
    logger.setLevel(logging.DEBUG)

    if logger.handlers:
        return logger

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [%(name)s:%(lineno)d] - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    logger.info(f"Initialized model evaluation pipeline logger. Log file: {log_file}")
    
    return logger
