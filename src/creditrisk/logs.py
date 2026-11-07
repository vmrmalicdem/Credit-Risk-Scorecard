"""Structured logging: one JSON object per line on stdout, for the API and for uvicorn's own messages."""
import json
import logging
from datetime import UTC, datetime

RESERVED = set(logging.LogRecord("", 0, "", 0, "", (), None).__dict__) | {"message", "asctime", "fields"}


class JsonFormatter(logging.Formatter):
    def format(self, record):
        entry = {"ts": datetime.fromtimestamp(record.created, UTC).isoformat(timespec="milliseconds"),
                 "level": record.levelname, "logger": record.name, "message": record.getMessage()}
        entry.update(getattr(record, "fields", {}))
        if record.exc_info:
            entry["exception"] = self.formatException(record.exc_info)
        return json.dumps(entry, default=str)


def log_config(level="INFO"):
    handler = {"class": "logging.StreamHandler", "formatter": "json", "stream": "ext://sys.stdout"}
    return {"version": 1, "disable_existing_loggers": False,
            "formatters": {"json": {"()": "creditrisk.logs.JsonFormatter"}}, "handlers": {"default": handler},
            "loggers": {"uvicorn": {"handlers": ["default"], "level": level, "propagate": False},
                        "uvicorn.error": {"level": level}, "uvicorn.access": {"handlers": [], "propagate": False},
                        "creditrisk": {"handlers": ["default"], "level": level, "propagate": False}}}
