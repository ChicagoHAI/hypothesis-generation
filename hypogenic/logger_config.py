import logging
import colorlog
from logging import Logger


class LoggerConfig:
    file_handler = None
    console_handler = None
    level = None

    @staticmethod
    def setup_logger(
        level=logging.DEBUG,
        log_file_path=None,
    ):
        LoggerConfig.level = level

        formatter = colorlog.ColoredFormatter(
            "%(asctime)s %(log_color)s[%(levelname)s] %(purple)s%(name)s: %(blue)s%(message)s",
            log_colors={
                "DEBUG": "cyan",
                "INFO": "green",
                "WARNING": "yellow",
                "ERROR": "red",
                "CRITICAL": "red,bg_white",
            },
        )

        if log_file_path is not None:
            LoggerConfig.file_handler = logging.FileHandler(log_file_path)
            LoggerConfig.file_handler.setLevel(logging.DEBUG)
            LoggerConfig.file_handler.setFormatter(formatter)

        LoggerConfig.console_handler = logging.StreamHandler()
        LoggerConfig.console_handler.setLevel(logging.DEBUG)
        LoggerConfig.console_handler.setFormatter(formatter)

    @staticmethod
    def get_logger(name: str) -> Logger:
        if LoggerConfig.console_handler is None:
            LoggerConfig.setup_logger()

        logger = logging.getLogger(name)
        logger.setLevel(LoggerConfig.level)
        logger.addHandler(LoggerConfig.console_handler)
        if LoggerConfig.file_handler is not None:
            logger.addHandler(LoggerConfig.file_handler)
        return logger
