"""日志工具。"""
import logging
import sys
from datetime import datetime


def get_logger(name: str = "advertiser_adaptive", level: int = logging.INFO) -> logging.Logger:
    """获取统一格式的 logger。"""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(level)
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)
    fmt = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(fmt)
    logger.addHandler(handler)
    return logger
