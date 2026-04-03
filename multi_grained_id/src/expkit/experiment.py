"""
实验配置和状态数据类

定义 ExperimentConfig 和 ExperimentState 数据类，用于实验管理。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class ExperimentConfig:
    """实验配置数据类"""
    exp_id: str
    name: str
    script: str           # 相对于 project_root 的路径
    result_dir: str       # 相对于 results/ 的路径
    priority: int
    description: str
    completed: bool = False
    use_gpu: bool = True
    max_restarts: int = 3
    min_memory_mb: int = 4096
    tags: List[str] = field(default_factory=list)


@dataclass
class ExperimentState:
    """实验运行状态数据类"""
    status: str = "pending"   # pending | running | completed | failed
    pid: Optional[int] = None
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    restart_count: int = 0
    last_error: Optional[str] = None
    updated_at: Optional[str] = None
