import os
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
        "<cyan>{name}</cyan> | "
        f"{tag}"
        "<level>{message}</level>\n"
    )


def _create_azure_monitor_sink(connection_string: str):
    """Create a loguru sink function that exports logs to Azure Monitor
    Application Insights as OpenTelemetry trace records.

    Each log record is sent with custom dimensions extracted from
    loguru's ``extra`` dict (component, request_id, video_id, state,
    tool_name, query) so they can be filtered in KQL.
    """
    from opentelemetry.sdk._logs import LoggerProvider
    from opentelemetry.sdk._logs.export import BatchLogRecordProcessor
    from azure.monitor.opentelemetry.exporter import AzureMonitorLogExporter
    from opentelemetry._logs import set_logger_provider, SeverityNumber
    from opentelemetry.sdk.resources import Resource

    resource = Resource.create({"service.name": "mmctagent-mcp"})
    provider = LoggerProvider(resource=resource)
    exporter = AzureMonitorLogExporter(connection_string=connection_string)
    provider.add_log_record_processor(BatchLogRecordProcessor(exporter))
    set_logger_provider(provider)
    otel_logger = provider.get_logger("mmctagent")

    _LEVEL_MAP = {
        "TRACE": SeverityNumber.TRACE,
        "DEBUG": SeverityNumber.DEBUG,
        "INFO": SeverityNumber.INFO,
        "SUCCESS": SeverityNumber.INFO2,
        "WARNING": SeverityNumber.WARN,
        "ERROR": SeverityNumber.ERROR,
        "CRITICAL": SeverityNumber.FATAL,
    }

    _EXTRA_KEYS = ("component", "request_id", "video_id", "state", "tool_name", "query")

    def _sink(message):
        record = message.record
        severity = _LEVEL_MAP.get(record["level"].name, SeverityNumber.INFO)

        attributes = {
            "logger.name": record["name"],
            "code.filepath": str(record["file"].path) if record["file"] else "",
            "code.lineno": record["line"],
            "code.function": record["function"],
        }
        for key in _EXTRA_KEYS:
            val = record["extra"].get(key)
            if val is not None:
                attributes[key] = str(val)

        otel_logger.emit(
            timestamp=int(record["time"].timestamp() * 1e9),
            severity_number=severity,
            severity_text=record["level"].name,
            body=str(record["message"]),
            attributes=attributes,
        )

    return _sink, provider


class LoggerManager:
    def __init__(self):
        self.console_sink_id = None
        self._azure_sink_id = None
        self._otel_provider = None

        # Always remove the default handler
        logger.remove()

    def enable_console(self, level: str = ""):
        """Enable console logging.

        Args:
            level: Log level override. If empty, reads ``LOG_LEVEL``
                   env var, defaulting to ``INFO``.
        """
        if self.console_sink_id is None:
            resolved_level = level or os.environ.get("LOG_LEVEL", "INFO")
            # diagnose=False prevents loguru from dumping frame locals on
            # exception logging. Frame locals can hold per-request secrets
            # (e.g., the ACL user_identifier_context dict carrying a graph
            # token); backtrace=True keeps the stack itself.
            self.console_sink_id = logger.add(
                sys.stdout,
                level=resolved_level,
                colorize=True,
                format=_format,
                backtrace=True,
                diagnose=False,
            )

    def disable_console(self):
        if self.console_sink_id is not None:
            logger.remove(self.console_sink_id)
            self.console_sink_id = None

    def enable_azure_monitor(self, connection_string: str = "", level: str = "DEBUG"):
        """Enable Azure Monitor Application Insights log export.

        Sends all logs at *level* (default DEBUG) to App Insights as
        structured traces with custom dimensions.

        Args:
            connection_string: App Insights connection string. If empty,
                reads ``APPLICATIONINSIGHTS_CONNECTION_STRING`` env var.
            level: Minimum level for the Azure sink.
        """
        if self._azure_sink_id is not None:
            return

        conn_str = connection_string or os.environ.get("APPLICATIONINSIGHTS_CONNECTION_STRING", "")
        if not conn_str:
            return

        try:
            sink_fn, provider = _create_azure_monitor_sink(conn_str)
            self._otel_provider = provider
            # diagnose=False — see enable_console() for rationale.
            self._azure_sink_id = logger.add(
                sink_fn,
                level=level,
                format="{message}",
                backtrace=True,
                diagnose=False,
            )
        except Exception as e:
            logger.warning(f"Failed to initialise Azure Monitor sink: {e}")

    def disable_azure_monitor(self):
        if self._azure_sink_id is not None:
            logger.remove(self._azure_sink_id)
            self._azure_sink_id = None
        if self._otel_provider is not None:
            self._otel_provider.shutdown()
            self._otel_provider = None

    def get_logger(self):
        return logger


log_manager = LoggerManager()
