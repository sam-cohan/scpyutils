"""
Utilities for logging.

Author: Sam Cohan
"""
import logging
import os
import re
from typing import Optional


def setup_logger(
    logger_name: str,
    file_name: Optional[str] = None,
    log_to_stdout: bool = True,
    log_level: int = logging.DEBUG,
    base_dir: str = "./logs",
):
    """Set up a logger which optionally also logs to file.

    Args:
        logger_name: name of logger. Note that if you set up a logger
            with a previously used name, you will simply change properties of
            the existing logger, so be careful!
        file_name: name of logging file. If nothing provided, will not
            log to file
        log_to_std_out: whether the log should be output to stdout
            (default: True)
        log_level: log levels from logging library (default:
            `logging.DEBUG`)
        base_dir: directory of where to put the log file (default:
            "./log")

    Returns:
        logger object.
    """
    assert file_name or log_to_stdout, "logger without output is useless!"

    _logger = logging.getLogger(logger_name)

    formatter_str = (
        "[%(asctime)s.%(msecs)03d: %(levelname)s]"
        " %(thread)d::%(filename)s::%(lineno)d"
        " ::%(name)s.%(funcName)s(): %(message)s"
    )
    time_format_str = "%Y-%m-%d %H:%M:%S"

    if file_name:
        # make sure base_dir is the full dir and file_name is just the filename
        log_path = os.path.join(base_dir, file_name)
        file_name = os.path.basename(log_path)
        base_dir = os.path.dirname(log_path)
        # creat the full directory if it does not exist
        if not os.path.exists(base_dir):
            os.makedirs(base_dir)
        file_handlers = [
            handler
            for handler in _logger.handlers
            if isinstance(handler, logging.FileHandler)
        ]
        if not file_handlers:
            file_handler = logging.FileHandler(log_path, mode="a")
            # set the handler log level to DEBUG so it can be controlled at logger level
            file_handler.setLevel(logging.DEBUG)
            formatter = logging.Formatter(  # pylint: disable=invalid-name
                formatter_str, time_format_str
            )
            file_handler.setFormatter(formatter)
            _logger.addHandler(file_handler)
    if log_to_stdout:
        stream_handlers = [
            handler
            for handler in _logger.handlers
            if type(handler) is logging.StreamHandler
        ]
        if not stream_handlers:
            console_handler = logging.StreamHandler()  # pylint: disable=invalid-name
            # set the handler log level to DEBUG so it can be controlled at logger level
            console_handler.setLevel(logging.DEBUG)
            try:
                import colorlog

                formatter_class = colorlog.ColoredFormatter
                formatter_str = "%(log_color)s" + formatter_str
            except:  # noqa: E722
                formatter_class = logging.Formatter
            formatter = formatter_class(
                formatter_str,
                time_format_str,
            )
            console_handler.setFormatter(formatter)
            _logger.addHandler(console_handler)

    _logger.setLevel(log_level)

    return _logger


def set_all_logger_levels(
    log_level: int,
    inc_regex: Optional[str] = None,
    exc_regex: Optional[str] = None,
):
    """Change logging level for all named loggers.

    Args:
        log_level: integer log level from `logging` module. positive values mean
            log level should be at least as aggressive and negative values mean
            log level should be at least as conservative.
        inc_regex: regular expression to apply as include filter for logger name.
        exc_regex: regular expression to apply as exclude filter for logger name.
    """
    more_logging = log_level > 0
    log_level = abs(log_level)
    loggers = [
        logging.getLogger(name)
        for name in logging.root.manager.loggerDict
        if (inc_regex is None or re.search(inc_regex, name))
        and (exc_regex is None or not re.search(exc_regex, name))
    ]
    for logger in loggers:
        cur_level = logger.getEffectiveLevel()
        if log_level < cur_level and more_logging:
            logger.setLevel(log_level)
        elif log_level > cur_level and not more_logging:
            logger.setLevel(log_level)
