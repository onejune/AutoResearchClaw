"""
日志进度解析模块

从训练日志中解析进度信息。
"""

from __future__ import annotations

import re
from pathlib import Path


class LogProgressParser:
    """从训练日志中解析进度信息"""

    def parse(self, log_file: Path) -> str:
        """
        从日志文件中解析进度字符串

        Args:
            log_file: 日志文件路径

        Returns:
            进度字符串，如 '54.3% (3467/6378)' 或 'loss=0.2720' 或 ''
        """
        if not log_file.exists():
            return ""

        try:
            with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()

            # 检查最后 100 行
            for line in reversed(lines[-100:]):
                # 匹配 Epoch 进度：Epoch 1/1: 3467/6378 [...]
                if 'Epoch 1/1:' in line and '|' in line:
                    match = re.search(r'(\d+)/(\d+)', line)
                    if match:
                        current = int(match.group(1))
                        total = int(match.group(2))
                        percent = (current / total * 100) if total > 0 else 0
                        return f"{percent:.1f}% ({current}/{total})"

                # 匹配 loss 值
                if 'loss=' in line:
                    match = re.search(r'loss=([0-9.]+)', line)
                    if match:
                        return f"loss={match.group(1)}"

            return ""
        except Exception:
            return ""
