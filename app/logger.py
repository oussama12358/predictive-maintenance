"""
app/logger.py
-------------
Structured JSON logging for production deployment.

In production (Docker/K8s), logs are consumed by external systems:
  - Datadog, ELK Stack, CloudWatch, Grafana Loki, etc.

JSON-structured logs enable:
  - Filtering by machine_id, risk_level, or prediction_latency
  - Alerting rules based on High-risk prediction frequency
  - Audit trails for compliance

Usage:
    from app.logger import get_logger
    logger = get_logger(__name__)
    logger.info("Prediction made", extra={"machine_id": "MCH-0001", "risk": "High"})
"""

import logging
import json
import sys
from datetime import datetime, timezone


class JSONFormatter(logging.Formatter):
    """
    Formats log records as single-line JSON objects.
    Each field is queryable in log aggregation systems.
    """

    def format(self, record: logging.LogRecord) -> str:
        log_object = {
            "timestamp":  datetime.now(timezone.utc).isoformat(),
            "level":      record.levelname,
            "logger":     record.name,
            "message":    record.getMessage(),
            "module":     record.module,
            "function":   record.funcName,
            "line":       record.lineno,
        }

        # Merge any extra fields passed via logger.info(..., extra={...})
        for key, value in record.__dict__.items():
            if key not in (
                "name", "msg", "args", "levelname", "levelno",
                "pathname", "filename", "module", "exc_info", "exc_text",
                "stack_info", "lineno", "funcName", "created", "msecs",
                "relativeCreated", "thread", "threadName", "processName",
                "process", "message", "taskName",
            ):
                log_object[key] = value

        # Attach exception info if present
        if record.exc_info:
            log_object["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_object, default=str)


def get_logger(name: str) -> logging.Logger:
    """
    Returns a named logger with JSON formatting pointing to stdout.

    Args:
        name: Logger name, typically __name__ of the calling module.

    Returns:
        Configured logging.Logger instance.
    """
    logger = logging.getLogger(name)

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(JSONFormatter())
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False  # Prevent duplicate logs in uvicorn

    return logger