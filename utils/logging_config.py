import logging
import sys
import os
from logging.handlers import RotatingFileHandler

def setup_logging():
    # Determine environment
    env = os.getenv('ENVIRONMENT', 'development').lower()
    
    # Set appropriate log levels based on environment
    if env == 'production':
        file_log_level = logging.INFO
        console_log_level = logging.WARNING
    else:
        file_log_level = logging.DEBUG
        console_log_level = logging.INFO
    
    # Create formatters
    detailed_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
    )
    
    # Ensure logs directory exists before creating handlers
    os.makedirs('logs', exist_ok=True)
    
    # Set up rotating file handler for detailed logging (10MB per file, keep 5 backup files)
    file_handler = RotatingFileHandler(
        'logs/detailed_matching.log',
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5
    )
    file_handler.setLevel(file_log_level)
    file_handler.setFormatter(detailed_formatter)
    
    # Set up console handler for important info
    console_handler = logging.StreamHandler()
    console_handler.setLevel(console_log_level)
    console_handler.setFormatter(detailed_formatter)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    
    # Remove any existing handlers to prevent duplication
    root_logger.handlers = []
    
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    
    # Set up specific loggers with propagate=False to prevent duplication
    loggers = ['matcher', 'llm_analyzer', 'pdf_processor', 'app']
    for logger_name in loggers:
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.DEBUG)
        logger.propagate = False  # Prevent message propagation to avoid duplication
        logger.handlers = []  # Clear any existing handlers
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

# Call this function at startup
setup_logging()
