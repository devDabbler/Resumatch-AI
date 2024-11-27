import logging
import sys

def setup_logging():
    # Create formatters
    detailed_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Set up file handler for detailed logging
    file_handler = logging.FileHandler('detailed_matching.log')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(detailed_formatter)
    
    # Set up console handler for important info
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
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
