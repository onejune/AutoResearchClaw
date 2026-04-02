"""
命令行接口模块

提供实验管理器的 CLI 入口。
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from .manager import ExperimentManager


def main(project_root: Path = None) -> None:
    """
    CLI 主函数

    Args:
        project_root: 项目根目录路径（如果为 None，则使用当前目录）
    """
    if project_root is None:
        project_root = Path.cwd()

    parser = argparse.ArgumentParser(description="实验管理器")
    parser.add_argument("--daemon", action="store_true", help="运行守护进程模式")
    parser.add_argument("--interval", type=int, default=30, help="守护进程检查间隔 (秒)")
    parser.add_argument("--status", action="store_true", help="只显示当前状态")
    parser.add_argument("--start", type=str, help="启动指定实验")
    parser.add_argument("--stop", type=str, help="停止指定实验")
    parser.add_argument("--restart", type=str, help="重启指定实验")

    args = parser.parse_args()

    manager = ExperimentManager(project_root)

    if args.status:
        manager.show_status()

    elif args.start:
        # 使用调度器自动分配资源
        exp_config = manager.experiments.get(args.start)
        if not exp_config:
            print(f"未知实验：{args.start}")
            return

        assignment = manager.scheduler.get_assignment(exp_config)
        if assignment:
            if assignment["use_gpu"]:
                manager.start_experiment(args.start, use_gpu=True, gpu_idx=assignment["gpu_idx"])
            else:
                manager.start_experiment(args.start, use_gpu=False)
        else:
            print("❌ 没有空闲资源")

    elif args.stop:
        manager.stop_experiment(args.stop)

    elif args.restart:
        manager.stop_experiment(args.restart)
        time.sleep(2)

        exp_config = manager.experiments.get(args.restart)
        if exp_config:
            assignment = manager.scheduler.get_assignment(exp_config)
            if assignment:
                if assignment["use_gpu"]:
                    manager.start_experiment(args.restart, use_gpu=True, gpu_idx=assignment["gpu_idx"])
                else:
                    manager.start_experiment(args.restart, use_gpu=False)
            else:
                print("❌ 没有空闲资源")

    elif args.daemon:
        manager.run_daemon(interval=args.interval)

    else:
        manager.show_status()
        print("\n🚀 执行一次自动调度...")
        manager.monitor_and_restart()
        manager.auto_schedule()


if __name__ == "__main__":
    main()
