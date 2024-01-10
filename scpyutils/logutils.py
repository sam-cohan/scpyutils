"""
Utilities for logging.

Author: Sam Cohan
"""
import logging
import os
import re
import time
from collections import OrderedDict
from typing import Any, Dict, Optional, Union

import colorlog
from pythonjsonlogger import jsonlogger

DEFAULT_FMT = (
    "[%(asctime)s: %(levelname)s]"
    " %(name)s::%(filename)s::%(lineno)d"
    "::%(module)s.%(funcName)s(): %(message)s"
)
DEFAULT_DATE_FMT = "%Y-%m-%dT%H:%M:%S.%f%z"


class TimeFormatterMixin:
    def formatTime(
        self,
        record: logging.LogRecord,
        datefmt: Optional[str] = None,
    ) -> str:
        """Mixin to handle formatting the %f for milliseconds."""
        assert isinstance(self, logging.Formatter)
        # Create the time structure from the timestamp
        record_time = self.converter(record.created)
        datefmt = datefmt or DEFAULT_DATE_FMT
        if datefmt and "%f" in datefmt:
            datefmt = datefmt.replace("%f", "xxx")
            time_formatted = time.strftime(datefmt, record_time)
            millis = int(record.msecs)
            return time_formatted.replace("xxx", f"{millis:03d}")
        return time.strftime(datefmt, record_time)


class CustomLogFormatter(TimeFormatterMixin, logging.Formatter):
    pass


class CustomColoredFormatter(TimeFormatterMixin, colorlog.ColoredFormatter):
    pass


class CustomJsonFormatter(TimeFormatterMixin, jsonlogger.JsonFormatter):
    """Custom JSON formatter for logging to Datadog."""

    def add_fields(
        self,
        log_record: Dict[str, Any],
        record: logging.LogRecord,
        message_dict: Dict[str, Any],
    ) -> None:
        super().add_fields(log_record, record, message_dict)
        custom_log_record: Dict[str, Any] = OrderedDict()
        custom_log_record["timeMillis"] = int(record.created * 1000)
        custom_log_record["level"] = record.levelname
        custom_log_record["ddsource"] = record.name
        message_data: Dict[str, Any] = {
            "message": record.getMessage(),
            "parameters": {
                "thread": record.thread,
                "filename": record.filename,
                "lineno": record.lineno,
                "name": record.name,
                "funcName": record.funcName,
                "module": record.module,
                "timestamp": self.formatTime(record, self.datefmt),
            },
        }
        if hasattr(record, "params"):
            arbitrary_params = getattr(record, "params")
            if isinstance(arbitrary_params, dict):
                message_data["parameters"].update(arbitrary_params)

        if record.exc_info:
            message_data["exception"] = self.formatException(record.exc_info)

        custom_log_record["messageData"] = message_data

        log_record.clear()
        log_record.update(custom_log_record)


def setup_logger(  # noqa: C901
    logger_name: str,
    log_to_console: Optional[bool] = None,
    log_to_file: Optional[bool] = None,
    level: int = logging.DEBUG,
    base_dir: Optional[str] = None,
    force_format: bool = True,
    log_as_json: Optional[bool] = None,
    fmt: str = DEFAULT_FMT,
    datefmt: str = DEFAULT_DATE_FMT,
) -> logging.Logger:
    if log_to_console is None:
        log_to_console = get_bool(os.environ.get("LOG_TO_CONSOLE", "1"))
    if log_to_file is None:
        log_to_file = get_bool(os.environ.get("LOG_TO_FILE", "1"))
    if log_as_json is None:
        log_as_json = get_bool(os.environ.get("LOG_AS_JSON", "0"))
    if force_format is None:
        force_format = get_bool(os.environ.get("LOG_FORCE_FORMAT", "1"))
    if base_dir is None:
        base_dir = os.environ.get("LOG_BASE_DIR", "./logs")
    assert log_to_file or log_to_console, "logger without output is useless!"
    _logger = logging.getLogger(logger_name)

    if log_to_file:
        file_handlers = [
            handler
            for handler in _logger.handlers
            if isinstance(handler, logging.FileHandler)
        ]
        if not file_handlers:
            # Make sure base_dir is the full dir and file_name is just the filename.
            log_path = os.path.join(base_dir, logger_name + ".log")
            base_dir = os.path.dirname(log_path)
            # Create the full directory if it does not exist.
            if not os.path.exists(base_dir):
                os.makedirs(base_dir)
            file_handler = logging.FileHandler(log_path, mode="a")
            file_handler.setLevel(logging.DEBUG)
            set_handler_formatter(
                handler=file_handler,
                fmt=None if log_as_json else fmt,
                datefmt=datefmt,
            )
            _logger.addHandler(file_handler)

    if log_to_console:
        stream_handlers = [
            handler
            for handler in _logger.handlers
            if type(handler) is logging.StreamHandler
        ]
        if not stream_handlers:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.DEBUG)
            set_handler_formatter(
                handler=console_handler,
                fmt=None if log_as_json else fmt,
                datefmt=datefmt,
            )
            _logger.addHandler(console_handler)
    if force_format:
        for handler in _logger.handlers:
            set_handler_formatter(
                handler=handler,
                fmt=None if log_as_json else fmt,
                datefmt=datefmt,
            )
    _logger.setLevel(level)
    return _logger


def set_handler_formatter(
    handler: logging.Handler,
    fmt: Optional[str] = DEFAULT_FMT,
    datefmt: str = DEFAULT_DATE_FMT,
) -> None:
    """Set the formatter for the given handler.

    Args:
        handler: The handler to set the formatter for.
        fmt: The format string for the formatter (default: DEFAULT_FMT)
        datefmt: The format string for the timestamp (default: DEFAULT_DATE_FMT).
    """
    if fmt is None:
        formatter: Union[logging.Formatter, CustomJsonFormatter] = CustomJsonFormatter(
            datefmt=datefmt
        )
    else:
        if isinstance(handler, logging.StreamHandler):
            formatter = CustomColoredFormatter(
                fmt=f"%(log_color)s{fmt}",
                datefmt=datefmt,
            )
        else:
            formatter = CustomLogFormatter(fmt=fmt, datefmt=datefmt)
    handler.setFormatter(formatter)


def set_all_logger_levels(
    level: int,
    inc_regex: Optional[str] = None,
    exc_regex: Optional[str] = None,
) -> None:
    """Set all logger levels.

    Args:
        level: The level to set the loggers to.
        inc_regex: Include only loggers matching this regex.
        exc_regex: Exclude loggers matching this regex.
    """
    more_logging = level > 0
    level = abs(level)
    loggers = [
        logging.getLogger(name)
        for name in logging.root.manager.loggerDict
        if (inc_regex is None or re.search(inc_regex, name))
        and (exc_regex is None or not re.search(exc_regex, name))
    ]
    for logger in loggers:
        cur_level = logger.getEffectiveLevel()
        if level < cur_level and more_logging:
            logger.setLevel(level)
        elif level > cur_level and not more_logging:
            logger.setLevel(level)


def set_all_logger_handlers_as_json_if_needed(
    inc_regex: Optional[str] = None,
    exc_regex: Optional[str] = None,
    datefmt: str = DEFAULT_DATE_FMT,
) -> None:
    """Set all logger handlers as JSON if needed.

    Will change the formatter of all handlers of all loggers to JSON if the
    environment variable LOG_AS_JSON is set to a truthy value.

    Args:
        inc_regex: Include only loggers matching this regex.
        exc_regex: Exclude loggers matching this regex.
        datefmt: The format string for the timestamp (default: DEFAULT_DATE_FMT).

    """
    if get_bool(os.environ.get("LOG_AS_JSON")):
        loggers = [
            logging.getLogger(name)
            for name in logging.root.manager.loggerDict
            if (inc_regex is None or re.search(inc_regex, name))
            and (exc_regex is None or not re.search(exc_regex, name))
        ]
        for logger in loggers:
            json_formatter = CustomJsonFormatter(datefmt=datefmt)
            for handler in logger.handlers:
                handler.setFormatter(json_formatter)


def get_bool(s: Optional[Union[str, bool]]) -> bool:
    """Convert a string to a boolean.

    Values that are considered True: "true", "1", "yes".

    Args:
        s: The string to convert.
    """
    if isinstance(s, bool):
        return s
    if s is None:
        return False
    return s.lower() in {"true", "1", "yes"}
