from loguru import logger
import sys
import os

class LoggerManager:
    def __init__(self):
        self.console_sink_id = None

        # Always remove the default handler
        logger.remove()

    def enable_console(self):
        if self.console_sink_id is None:
            self.console_sink_id = logger.add(sys.stdout, level="INFO", colorize=True)

    def disable_console(self):
        if self.console_sink_id is not None:
            logger.remove(self.console_sink_id)
            self.console_sink_id = None

    def get_logger(self):
        return logger

log_manager = LoggerManager()

if __name__=="__main__":
    logger = log_manager.get_logger()
    logger.info("Info log level 1")
    logger.error("Error log level ")
    log_manager.enable_console()
    logger.info("Info log level 2")
    logger.error("Error log level 2")
    log_manager.disable_console()
    logger.info("Info log level 3")
    logger.error("Error log level 3")
    log_manager.enable_console()
    logger.info("Info log level 4")
    logger.error("Error log level 4")