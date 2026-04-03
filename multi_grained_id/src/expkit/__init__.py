"""
expkit - 实验管理工具包

提供硬件监控、实验配置管理和自动调度功能。
"""

from .hardware import HardwareMonitor, DeviceInfo, DeviceType, select_training_device
from .experiment import ExperimentConfig, ExperimentState
from .manager import ExperimentManager
from .scheduler import ResourceScheduler
from .progress import LogProgressParser
from .config import ExpKitConfig

__version__ = "2.0.0"
__all__ = [
    "HardwareMonitor",
    "DeviceInfo",
    "DeviceType",
    "select_training_device",
    "ExperimentConfig",
    "ExperimentState",
    "ExperimentManager",
    "ResourceScheduler",
    "LogProgressParser",
    "ExpKitConfig",
]
