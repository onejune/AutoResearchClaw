"""
expkit 配置加载模块

从项目根目录的 expkit.conf 读取集中配置，所有路径均相对于项目根目录解析。
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List


# expkit.conf 中的默认值（与文件保持一致）
_DEFAULTS = {
    "PROJECT_NAME": "",
    "EXPERIMENTS_YAML": "experiments_config.yaml",
    "EXPERIMENTS_JSON": "experiments_config.json",
    "DAEMON_LOG": "logs/daemon.log",
    "DAEMON_PID": "logs/daemon.pid",
    "EXP_LOGS_DIR": "logs/experiments",
    "STATE_FILE": "experiment_manager_state.json",
    "RESULTS_DIR": "results",
    "DAEMON_INTERVAL": "30",
    "MAX_RESTARTS": "3",
    "GPU_IDLE_UTIL": "10",
    "GPU_IDLE_MEM_MB": "500",
    "CPU_IDLE_UTIL": "70",
    "COMPLETION_KEYS": "auc,AUC,accuracy,loss_final",
}


def _parse_conf(conf_file: Path) -> dict:
    """解析 key=value 格式的配置文件，忽略注释和空行"""
    result = {}
    if not conf_file.exists():
        return result
    with open(conf_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                key, _, value = line.partition("=")
                result[key.strip()] = value.strip()
    return result


class ExpKitConfig:
    """
    实验管理系统配置

    优先级：expkit.conf > 内置默认值
    所有路径属性均返回绝对 Path 对象。
    """

    def __init__(self, project_root: Path) -> None:
        self.project_root = project_root.resolve()
        raw = {**_DEFAULTS, **_parse_conf(project_root / "expkit.conf")}

        # 项目标识：优先用配置，否则用目录名
        self.project_name: str = raw["PROJECT_NAME"] or self.project_root.name

        # 配置文件路径
        self.experiments_yaml: Path = self.project_root / raw["EXPERIMENTS_YAML"]
        self.experiments_json: Path = self.project_root / raw["EXPERIMENTS_JSON"]

        # 运行时文件路径
        self.daemon_log: Path = self.project_root / raw["DAEMON_LOG"]
        self.daemon_pid: Path = self.project_root / raw["DAEMON_PID"]
        self.exp_logs_dir: Path = self.project_root / raw["EXP_LOGS_DIR"]
        self.state_file: Path = self.project_root / raw["STATE_FILE"]

        # 结果目录
        self.results_dir: Path = self.project_root / raw["RESULTS_DIR"]

        # 守护进程参数
        self.daemon_interval: int = int(raw["DAEMON_INTERVAL"])
        self.max_restarts: int = int(raw["MAX_RESTARTS"])

        # 资源调度阈值
        self.gpu_idle_util: float = float(raw["GPU_IDLE_UTIL"])
        self.gpu_idle_mem_mb: float = float(raw["GPU_IDLE_MEM_MB"])
        self.cpu_idle_util: float = float(raw["CPU_IDLE_UTIL"])

        # 完成检测字段
        self.completion_keys: List[str] = [
            k.strip() for k in raw["COMPLETION_KEYS"].split(",") if k.strip()
        ]

    def ensure_dirs(self) -> None:
        """确保所有运行时目录存在"""
        self.daemon_log.parent.mkdir(parents=True, exist_ok=True)
        self.exp_logs_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def __repr__(self) -> str:
        return (
            f"ExpKitConfig(project={self.project_name!r}, "
            f"root={self.project_root})"
        )
