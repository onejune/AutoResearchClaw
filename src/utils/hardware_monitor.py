"""
硬件监控模块 - 向后兼容包装

此文件保持向后兼容，从 expkit.hardware 导入所有功能。
"""

import sys
from pathlib import Path

# 添加 src 到路径以便导入 expkit
sys.path.insert(0, str(Path(__file__).parent.parent))

from expkit.hardware import HardwareMonitor, DeviceInfo, DeviceType, select_training_device

__all__ = ["HardwareMonitor", "DeviceInfo", "DeviceType", "select_training_device"]
