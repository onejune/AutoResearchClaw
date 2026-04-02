#!/usr/bin/env python3
"""实验管理器入口 - 使用 src/expkit 包"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from expkit.cli import main

if __name__ == "__main__":
    main(project_root=PROJECT_ROOT)
