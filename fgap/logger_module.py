# logger_module.py

import logging

from fgap_config import LOG_TO_FILE, LOG_TO_CONSOLE, LOG_FILE_NAME


def setup_logging():
    """Set up logging configuration based on macro definitions."""
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Create handlers if the respective macros are set to True
    if LOG_TO_FILE:
        file_handler = logging.FileHandler(LOG_FILE_NAME, mode='a')  # Append mode
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    if LOG_TO_CONSOLE:
        console_handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

def log_message(message):
    """Log a message to the configured outputs."""
    logging.info(message)