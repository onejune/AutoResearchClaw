"""
资源调度模块

从 ExperimentManager 中拆分出来的纯调度逻辑，不依赖项目路径。
"""

from __future__ import annotations

from typing import Dict, Optional

from .hardware import HardwareMonitor
from .experiment import ExperimentConfig


class ResourceScheduler:
    """资源调度器 - 纯逻辑，不依赖项目路径"""

    def __init__(self, hardware: HardwareMonitor) -> None:
        """
        初始化资源调度器

        Args:
            hardware: HardwareMonitor 实例
        """
        self.hardware = hardware

    def get_assignment(self, exp: ExperimentConfig) -> Optional[Dict]:
        """
        获取实验的资源分配方案

        Args:
            exp: 实验配置

        Returns:
            {"use_gpu": True, "gpu_idx": 0} 或 {"use_gpu": False} 或 None（无资源）
        """
        # 刷新硬件状态
        self.hardware.refresh()

        if exp.use_gpu:
            # 尝试分配 GPU
            idle_gpus = self.hardware.get_idle_gpus(max_util=10.0, max_mem_mb=500)
            if idle_gpus:
                return {"use_gpu": True, "gpu_idx": idle_gpus[0]}

            # GPU 不足时降级到 CPU
            if self.hardware.is_cpu_idle(max_util=70.0):
                return {"use_gpu": False}

            # 资源不足
            return None
        else:
            # 不需要 GPU，检查 CPU
            if self.hardware.is_cpu_idle(max_util=70.0):
                return {"use_gpu": False}

            # CPU 也不空闲
            return None

    def can_run_on_gpu(self, min_memory_mb: int = 4096) -> bool:
        """
        检查是否有可用 GPU

        Args:
            min_memory_mb: 最小显存需求 (MB)

        Returns:
            是否有满足条件的 GPU
        """
        self.hardware.refresh()
        for gpu in self.hardware.gpus:
            if (gpu.utilization < 10.0 and
                gpu.memory_used_mb < 500 and
                gpu.memory_available_mb >= min_memory_mb):
                return True
        return False

    def can_run_on_cpu(self) -> bool:
        """
        检查 CPU 是否可用

        Returns:
            CPU 是否空闲
        """
        return self.hardware.is_cpu_idle(max_util=70.0)
