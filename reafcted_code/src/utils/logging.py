# logging.py
import logging

def setup_logger(name: str, log_file: str, level=logging.INFO):
    """Function to set up a logger."""
    handler = logging.FileHandler(log_file)
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)

    return logger

# # Example usage:
# logger = setup_logger('nlp_pipeline', 'nlp_pipeline.log')
# logger.info('This is an informational message.')
