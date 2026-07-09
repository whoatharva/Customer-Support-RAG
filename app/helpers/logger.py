import logging
import sys


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


def configure_logging(level: str = "INFO"):
    fmt = "%(asctime)s [%(levelname)s] %(name)s — %(message)s"
    logging.basicConfig(stream=sys.stdout, level=level.upper(), format=fmt, force=True)
