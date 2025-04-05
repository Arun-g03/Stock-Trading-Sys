# logger.py

import logging
import os
import traceback

# Configuration for the logging directory
LOG_DIR = 'logs'
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)
LOG_PATH = os.path.join(LOG_DIR, 'system.log')

class System_Log:
    @staticmethod
    def setup_logger(name, log_file=LOG_PATH, level=logging.INFO):
        """
        Setup a logger with the specified name and log file.
        """
        handler = logging.FileHandler(log_file)
        handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s'))

        logger = logging.getLogger(name)
        logger.setLevel(level)
        logger.addHandler(handler)

        return logger

# Example usage:
# system_logger = System_Log.setup_logger('main_logger')
# system_logger.info('This is an informational message')
# system_logger.error('This is an error message')
