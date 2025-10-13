import sys
from loguru import logger

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