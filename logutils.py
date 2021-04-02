import logging
import os

import colorlog

logger_ = logging.getLogger(__name__)


def logger(msg, request_logging_levels=None, filter_logging_levels=None, severity=None):
    request_logging_levels = request_logging_levels or ["debug"]
    try:
        if not request_logging_levels:
            return
        if "debug" in request_logging_levels or (
            filter_logging_levels
            and (
                "debug" in filter_logging_levels
                or (any([x in filter_logging_levels for x in request_logging_levels]))
            )
        ):
            severity_log(msg, severity)
    except Exception as e:
        severity_log("Failed log: %s" % e, "warning")


def severity_log(msg, severity):
    severity = severity or "info"
    getattr(logger_, severity, "info")(msg)


def setup_logger(
    logger_name,
    file_name=None,
    log_to_stdout=True,
    log_level=logging.DEBUG,
    base_dir="./logs",
):
    """Set up a logger which optionally also logs to file.
    Args:
        logger_name (str): name of logger. Note that if you set up a logger
            with a previously used name, you will simply change properties of
            the existing logger, so be careful!
        file_name (str): name of logging file. If nothing provided, will not
            log to file
        log_to_std_out (bool): whether the log should be output to stdout
            (default: True)
        log_level (int): log levels from logging libarary (default:
            `logging.DEBUG`)
        base_dir (str): directory of where to put the log file (default:
            "./log")
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
            color_formatter = colorlog.ColoredFormatter(
                "%(log_color)s" + formatter_str,
                time_format_str,
            )
            console_handler.setFormatter(color_formatter)
            _logger.addHandler(console_handler)

    _logger.setLevel(log_level)

    return _logger


class ToggleHelper:
    def validate_toggle(self, toggle):
        if toggle:
            try:
                if not self.log_toggles[toggle]:
                    return False
            except:
                pass
            self.log_toggles[toggle] = False
        return True
