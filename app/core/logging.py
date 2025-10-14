"""Logging configuration utilities."""

import logging
import logging.config
from typing import Optional

from app.core.config import settings


def get_logging_config(level: Optional[str] = None, json_logs: Optional[bool] = None) -> dict:
    """Return a dictConfig-compatible logging configuration."""

    log_level = (level or settings.LOG_LEVEL).upper()
    use_json = json_logs if json_logs is not None else settings.LOG_JSON

    formatter = (
        {
            "format": "%(message)s",
            "class": "pythonjsonlogger.jsonlogger.JsonFormatter",
        }
        if use_json
        else {
            "format": "%(levelname)s | %(asctime)s | %(name)s | %(message)s",
            "datefmt": "%Y-%m-%d %H:%M:%S",
        }
    )

    config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "default": formatter,
        },
        "handlers": {
            "default": {
                "class": "logging.StreamHandler",
                "formatter": "default",
                "stream": "ext://sys.stdout",
            }
        },
        "loggers": {
            "": {
                "handlers": ["default"],
                "level": log_level,
            },
            "uvicorn": {"handlers": ["default"], "level": "INFO", "propagate": False},
            "uvicorn.error": {"handlers": ["default"], "level": "INFO", "propagate": False},
            "uvicorn.access": {"handlers": ["default"], "level": "INFO", "propagate": False},
        },
    }

    return config


def configure_logging(level: Optional[str] = None, json_logs: Optional[bool] = None) -> None:
    """Configure logging for the application."""

    logging_config = get_logging_config(level=level, json_logs=json_logs)
    logging.config.dictConfig(logging_config)

