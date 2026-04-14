import sys
from loguru import logger


def _format(record: dict) -> str:
    """Build a loguru format string with optional component tag.

    When code uses ``logger.bind(component="state")``, the log line
    includes a ``[state]`` tag between the level and the message.
    Without a bound component the tag is omitted.
    """
    component = record["extra"].get("component", "")
    tag = f"[{component}] " if component else ""
    return (
        "<green>{time:HH:mm:ss.SSS}</green> | "
        "<level>{level: <8}</level> | "
        f"{tag}"
        "<level>{message}</level>\n"
    )


class LoggerManager:
    def __init__(self):
        self.console_sink_id = None

        # Always remove the default handler
        logger.remove()

    def enable_console(self, level="INFO"):
        if self.console_sink_id is None:
            self.console_sink_id = logger.add(
                sys.stdout, level=level, colorize=True, format=_format,
            )

    def disable_console(self):
        if self.console_sink_id is not None:
            logger.remove(self.console_sink_id)
            self.console_sink_id = None

    def get_logger(self):
        return logger


log_manager = LoggerManager()
