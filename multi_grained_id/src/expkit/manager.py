"""
实验管理器模块

管理实验队列、监控运行状态、自动调度和重启。
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import psutil

from .hardware import HardwareMonitor
from .experiment import ExperimentConfig, ExperimentState
from .scheduler import ResourceScheduler
from .progress import LogProgressParser


# 默认日志目录
DEFAULT_LOGS_DIR = Path("/tmp/exp_logs")


class ExperimentManager:
    """实验管理器 - 管理实验队列和调度"""

    def __init__(self, project_root: Path) -> None:
        """
        初始化实验管理器

        Args:
            project_root: 项目根目录路径
        """
        self.project_root = project_root
        self.results_dir = project_root / "results"
        self.logs_dir = DEFAULT_LOGS_DIR
        self.state_file = project_root / "experiment_manager_state.json"
        self.config_file = project_root / "experiments_config.json"

        # 确保日志目录存在
        self.logs_dir.mkdir(parents=True, exist_ok=True)

        # 加载实验配置
        self.experiments: Dict[str, ExperimentConfig] = {}
        self._load_experiments_from_config()

        # 运行中的进程记录
        self.running_pids: Dict[str, int] = {}

        # 加载状态
        self.state: Dict = self._load_state()

        # 硬件监控器和调度器
        self.monitor = HardwareMonitor()
        self.scheduler = ResourceScheduler(self.monitor)
        self.progress_parser = LogProgressParser()

        # 设置信号处理器
        self._setup_signal_handlers()

    def _setup_signal_handlers(self) -> None:
        """设置信号处理器，优雅退出"""
        signal.signal(signal.SIGTERM, self._signal_handler)
        signal.signal(signal.SIGINT, self._signal_handler)

    def _signal_handler(self, signum: int, frame) -> None:
        """处理终止信号"""
        print(f"\n收到信号 {signum}, 正在清理...")
        self.stop_all()
        self._save_state()
        sys.exit(0)

    def _load_experiments_from_config(self) -> None:
        """从配置文件加载实验列表"""
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    config_data = json.load(f)

                for exp_id, exp_config in config_data.items():
                    # 确保 exp_id 字段存在（兼容旧格式）
                    if "exp_id" not in exp_config:
                        exp_config["exp_id"] = exp_id
                    self.experiments[exp_id] = ExperimentConfig(**exp_config)

                print(f"✅ 已加载 {len(self.experiments)} 个实验配置")
            except Exception as e:
                print(f"⚠️  加载配置文件失败：{e}，使用默认配置")
                self._init_default_experiments()
        else:
            print("⚠️  配置文件不存在，初始化默认配置")
            self._init_default_experiments()
            self._save_experiments_to_config()

    def _init_default_experiments(self) -> None:
        """初始化默认实验配置"""
        self.experiments = {
            "exp001": ExperimentConfig(
                exp_id="exp001",
                name="AutoEmb",
                script="scripts/run_exp001_autoemb_v3.py",
                result_dir="exp001_autoemb",
                priority=1,
                description="自动化维度分配",
                completed=False,
                use_gpu=True,
                max_restarts=3
            ),
            "exp002": ExperimentConfig(
                exp_id="exp002",
                name="DDS",
                script="scripts/run_exp002_dds.py",
                result_dir="exp002_dds",
                priority=2,
                description="数据分布搜索",
                completed=False,
                use_gpu=True,
                max_restarts=3
            ),
            "exp003": ExperimentConfig(
                exp_id="exp003",
                name="Hierarchical v2",
                script="scripts/run_exp003_hierarchical_v2.py",
                result_dir="exp003_hierarchical_v2",
                priority=3,
                description="层次化 Embedding (deep chain)",
                completed=True,
                use_gpu=True,
                max_restarts=3
            ),
            "exp004": ExperimentConfig(
                exp_id="exp004",
                name="MetaEmb",
                script="scripts/run_exp004_metaemb.py",
                result_dir="exp004_metaemb",
                priority=4,
                description="元学习冷启动",
                completed=False,
                use_gpu=False,
                max_restarts=3
            ),
            "exp005": ExperimentConfig(
                exp_id="exp005",
                name="Contrastive",
                script="scripts/run_exp005_contrastive.py",
                result_dir="exp005_contrastive",
                priority=5,
                description="对比学习增强",
                completed=False,
                use_gpu=False,
                max_restarts=3
            ),
            "exp006": ExperimentConfig(
                exp_id="exp006",
                name="FiBiNET",
                script="scripts/run_exp006_fibinet.py",
                result_dir="exp006_fibinet",
                priority=6,
                description="特征交互",
                completed=False,
                use_gpu=False,
                max_restarts=3
            ),
            "exp007": ExperimentConfig(
                exp_id="exp007",
                name="Combined",
                script="scripts/run_exp007_combined.py",
                result_dir="exp007_combined",
                priority=7,
                description="综合方案",
                completed=False,
                use_gpu=False,
                max_restarts=3
            ),
            "exp008": ExperimentConfig(
                exp_id="exp008",
                name="DeepFM",
                script="scripts/run_exp008_deepfm.py",
                result_dir="exp008_deepfm",
                priority=8,
                description="DeepFM 模型对比",
                completed=False,
                use_gpu=False,
                max_restarts=3
            )
        }

    def _save_experiments_to_config(self) -> None:
        """保存实验配置到文件"""
        config_data = {}
        for exp_id, exp in self.experiments.items():
            config_data[exp_id] = {
                "exp_id": exp.exp_id,
                "name": exp.name,
                "script": exp.script,
                "result_dir": exp.result_dir,
                "priority": exp.priority,
                "description": exp.description,
                "completed": exp.completed,
                "use_gpu": exp.use_gpu,
                "max_restarts": exp.max_restarts,
                "min_memory_mb": exp.min_memory_mb,
                "tags": exp.tags
            }

        with open(self.config_file, 'w', encoding='utf-8') as f:
            json.dump(config_data, f, indent=2, ensure_ascii=False)
        print(f"✅ 已保存实验配置到 {self.config_file}")

    def _load_state(self) -> Dict:
        """加载管理器状态"""
        if self.state_file.exists():
            try:
                with open(self.state_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception:
                pass

        return {
            "started_at": None,
            "updated_at": None,
            "experiments": {
                exp_id: {
                    "status": "pending",
                    "pid": None,
                    "start_time": None,
                    "end_time": None,
                    "restart_count": 0,
                    "last_error": None,
                    "updated_at": None
                }
                for exp_id in self.experiments
            }
        }

    def _save_state(self) -> None:
        """保存管理器状态"""
        self.state["updated_at"] = datetime.now().isoformat()
        with open(self.state_file, 'w', encoding='utf-8') as f:
            json.dump(self.state, f, indent=2, ensure_ascii=False)

    def is_experiment_running(self, exp_id: str) -> bool:
        """
        检查实验是否正在运行（通过脚本路径匹配）
        如果进程存在但状态不是 running，自动同步状态。

        Args:
            exp_id: 实验 ID

        Returns:
            实验是否正在运行
        """
        # 先检查记录的 PID
        if exp_id in self.running_pids:
            pid = self.running_pids[exp_id]
            if psutil.pid_exists(pid):
                try:
                    proc = psutil.Process(pid)
                    if proc.status() in (psutil.STATUS_RUNNING, psutil.STATUS_SLEEPING):
                        self._sync_running_status(exp_id)
                        return True
                except Exception:
                    pass

        # 通过脚本路径检查（更可靠）
        exp = self.experiments.get(exp_id)
        if not exp:
            return False

        script_path = self.project_root / exp.script

        for proc in psutil.process_iter(['pid', 'cmdline']):
            try:
                cmdline = proc.cmdline() or []
                cmdline_str = ' '.join(cmdline)
                # 匹配脚本路径而不是 exp_id 字符串
                if str(script_path) in cmdline_str and 'python' in cmdline_str:
                    self.running_pids[exp_id] = proc.pid
                    self._sync_running_status(exp_id)
                    return True
            except Exception:
                pass

        return False

    def _sync_running_status(self, exp_id: str) -> None:
        """如果进程实际在运行但状态不是 running，自动同步"""
        if exp_id not in self.state["experiments"]:
            return
        current = self.state["experiments"][exp_id].get("status")
        if current != "running":
            self.state["experiments"][exp_id]["status"] = "running"
            self.state["experiments"][exp_id]["updated_at"] = datetime.now().isoformat()
            self._save_state()

    def start_experiment(self, exp_id: str, use_gpu: bool = True, gpu_idx: int = None) -> bool:
        """
        启动实验

        Args:
            exp_id: 实验 ID
            use_gpu: 是否使用 GPU
            gpu_idx: GPU 索引（use_gpu=True 时有效）

        Returns:
            是否启动成功
        """
        if exp_id not in self.experiments:
            print(f"未知实验：{exp_id}")
            return False

        exp = self.experiments[exp_id]

        # 检查是否已在运行
        if self.is_experiment_running(exp_id):
            print(f"{exp_id} 已经在运行中")
            return True

        script_path = self.project_root / exp.script
        if not script_path.exists():
            print(f"脚本不存在：{script_path}")
            return False

        log_file = self.logs_dir / f"{exp_id}.log"

        print(f"🚀 启动 {exp_id} ({exp.name}): {exp.description}")
        print(f"   脚本：{exp.script}")
        print(f"   日志：{log_file}")
        if use_gpu and gpu_idx is not None:
            print(f"   GPU: {gpu_idx}")
        elif not use_gpu:
            print(f"   设备：CPU")

        try:
            # 构建环境变量
            env = os.environ.copy()
            if use_gpu and gpu_idx is not None:
                env["CUDA_VISIBLE_DEVICES"] = str(gpu_idx)
            else:
                env["CUDA_VISIBLE_DEVICES"] = ""

            # 直接用 Popen 启动，不用 shell=True + nohup，避免 pid 错乱
            with open(log_file, 'w') as lf:
                process = subprocess.Popen(
                    [sys.executable, "-u", str(script_path)],
                    cwd=str(self.project_root),
                    env=env,
                    stdout=lf,
                    stderr=lf,
                    start_new_session=True  # 脱离父进程组，不受 SIGHUP 影响
                )

            # 直接拿到真实 PID
            self.running_pids[exp_id] = process.pid
            self._update_experiment_status(exp_id, "running")
            print(f"   PID: {process.pid}")

            return True

        except Exception as e:
            print(f"启动 {exp_id} 失败：{e}")
            self._update_experiment_status(exp_id, "failed", error=str(e))
            return False

    def stop_experiment(self, exp_id: str) -> bool:
        """
        停止实验（先用 pid 精确 kill，再 pkill 兜底）

        Args:
            exp_id: 实验 ID

        Returns:
            是否停止成功
        """
        stopped = False

        # 方法 1：用记录的 pid 精确 kill
        if exp_id in self.running_pids:
            pid = self.running_pids[exp_id]
            if psutil.pid_exists(pid):
                try:
                    proc = psutil.Process(pid)
                    proc.terminate()
                    proc.wait(timeout=10)
                    print(f"✅ 已停止 {exp_id} (PID: {pid})")
                    del self.running_pids[exp_id]
                    stopped = True
                except Exception:
                    pass

        # 方法 2：通过脚本路径 pkill 兜底
        if not stopped:
            exp = self.experiments.get(exp_id)
            if exp:
                script_path = str(self.project_root / exp.script)
                try:
                    subprocess.run(["pkill", "-f", script_path], check=False)
                    print(f"✅ 已停止 {exp_id}")
                    stopped = True
                except Exception:
                    pass

        return stopped

    def stop_all(self) -> None:
        """停止所有实验"""
        for exp_id in list(self.running_pids.keys()):
            self.stop_experiment(exp_id)

    def _update_experiment_status(self, exp_id: str, status: str, error: str = None) -> None:
        """更新实验状态"""
        if exp_id not in self.state["experiments"]:
            return

        self.state["experiments"][exp_id]["status"] = status
        self.state["experiments"][exp_id]["updated_at"] = datetime.now().isoformat()

        if status == "running" and not self.state["experiments"][exp_id]["start_time"]:
            self.state["experiments"][exp_id]["start_time"] = datetime.now().isoformat()

        if status in ["completed", "failed"]:
            self.state["experiments"][exp_id]["end_time"] = datetime.now().isoformat()
            if error:
                self.state["experiments"][exp_id]["last_error"] = error

        self._save_state()

    def check_experiment_completed(self, exp_id: str) -> bool:
        """
        检查实验是否完成（检查 results.json 中是否有 auc 或 AUC 字段）

        Args:
            exp_id: 实验 ID

        Returns:
            实验是否已完成
        """
        exp = self.experiments.get(exp_id)
        if not exp:
            return False

        result_dir = self.results_dir / exp.result_dir
        results_json = result_dir / "results.json"

        if results_json.exists():
            try:
                with open(results_json, 'r') as f:
                    data = json.load(f)
                    if "auc" in data or "AUC" in data:
                        return True
            except Exception:
                pass

        return False

    def monitor_and_restart(self, max_restarts: int = 3) -> None:
        """监控实验并自动重启失败的"""
        for exp_id, exp_config in self.experiments.items():
            if exp_config.completed:
                continue

            state = self.state["experiments"].get(exp_id, {})
            current_status = state.get("status", "pending")
            actually_running = self.is_experiment_running(exp_id)

            # 进程实际在跑但状态不对 → 修正状态，跳过重启
            if actually_running and current_status in ("failed", "pending"):
                self._sync_running_status(exp_id)
                continue

            if current_status == "running" and not actually_running:
                restart_count = state.get("restart_count", 0)

                if restart_count < max_restarts:
                    print(f"⚠️  {exp_id} 意外停止，正在重启 (第 {restart_count + 1} 次)...")
                    self._update_experiment_status(exp_id, "failed", error="Process died unexpectedly")
                    time.sleep(5)

                    # 使用调度器分配资源
                    assignment = self.scheduler.get_assignment(exp_config)
                    if assignment:
                        if assignment["use_gpu"]:
                            self.start_experiment(exp_id, use_gpu=True, gpu_idx=assignment["gpu_idx"])
                        else:
                            self.start_experiment(exp_id, use_gpu=False)
                        self.state["experiments"][exp_id]["restart_count"] = restart_count + 1
                    else:
                        print(f"   资源不足，等待空闲...")
                else:
                    print(f"❌ {exp_id} 重启次数达到上限 ({max_restarts}), 放弃重启")
                    self._update_experiment_status(exp_id, "failed",
                                                  error=f"Max restarts ({max_restarts}) reached")

            elif current_status == "running" and actually_running:
                if self.check_experiment_completed(exp_id):
                    print(f"✅ {exp_id} 已完成!")
                    self._update_experiment_status(exp_id, "completed")
                    if exp_id in self.running_pids:
                        del self.running_pids[exp_id]

    def auto_schedule(self) -> None:
        """自动调度：根据资源情况启动待运行的实验"""
        pending_exps: List[tuple] = []
        for exp_id, exp_config in self.experiments.items():
            if exp_config.completed:
                continue

            state = self.state["experiments"].get(exp_id, {})
            status = state.get("status", "pending")

            if status != "running" and not self.is_experiment_running(exp_id):
                pending_exps.append((exp_id, exp_config.priority))

        pending_exps.sort(key=lambda x: x[1])

        # 刷新硬件状态
        self.monitor.refresh()
        idle_gpus = self.monitor.get_idle_gpus()
        cpu_idle = self.monitor.is_cpu_idle()

        print(f"\n📊 资源状态:")
        print(f"   空闲 GPU: {idle_gpus}")
        print(f"   CPU 空闲：{cpu_idle}")
        print(f"   待运行实验：{len(pending_exps)}")

        for exp_id, _ in pending_exps:
            exp_config = self.experiments[exp_id]
            assigned = False

            # 使用调度器分配资源
            assignment = self.scheduler.get_assignment(exp_config)
            if assignment:
                if assignment["use_gpu"]:
                    if self.start_experiment(exp_id, use_gpu=True, gpu_idx=assignment["gpu_idx"]):
                        assigned = True
                        print(f"   → {exp_id} 分配到 GPU:{assignment['gpu_idx']}")
                else:
                    if self.start_experiment(exp_id, use_gpu=False):
                        assigned = True
                        print(f"   → {exp_id} 分配到 CPU")
            else:
                print(f"   ⏸️  {exp_id} 等待资源...")
                break

    def show_status(self) -> None:
        """显示当前状态"""
        print("\n" + "="*80)
        print("📊 实验管理器状态")
        print("="*80)

        # 使用 HardwareMonitor 显示 GPU 状态
        self.monitor.refresh()
        print(f"\n🖥️  GPU 状态 ({len(self.monitor.gpus)} 个):")
        for gpu in self.monitor.gpus:
            status = "✅ 空闲" if gpu.utilization < 10 and gpu.memory_used_mb < 500 else "🔴 忙碌"
            print(f"   GPU {gpu.device_id}: {status} ({gpu.name})")
            print(f"      利用率：{gpu.utilization:.1f}%, 显存：{gpu.memory_used_mb:.0f}MB / {gpu.memory_total_mb:.0f}MB")

        if not self.monitor.gpus:
            print("   无可用 GPU")

        # CPU 状态
        cpu = self.monitor.cpu
        cpu_status = "✅ 空闲" if cpu.utilization < 70 else "🔴 忙碌"
        print(f"\n💻  CPU 状态：{cpu_status}")
        print(f"   利用率：{cpu.utilization:.1f}%")
        print(f"   内存：{cpu.memory_used_mb:.0f}MB / {cpu.memory_total_mb:.0f}MB")

        # 实验状态
        print("\n🧪 实验队列:")
        print("-" * 100)
        print(f"{'ID':<10} {'名称':<20} {'状态':<12} {'进度':<20} {'备注'}")
        print("-" * 100)

        for exp_id in sorted(self.experiments.keys()):
            exp = self.experiments[exp_id]
            state = self.state["experiments"].get(exp_id, {})

            if exp.completed:
                status = "✅ 已完成"
                progress = ""
            elif self.is_experiment_running(exp_id):
                status = "🔄 运行中"
                log_file = self.logs_dir / f"{exp_id}.log"
                progress = self.progress_parser.parse(log_file)
            elif state.get("status") == "completed":
                status = "✅ 已完成"
                progress = ""
            elif state.get("status") == "failed":
                status = "❌ 失败"
                progress = state.get("last_error", "")[:30]
            else:
                status = "⏸️  待运行"
                progress = ""

            progress_str = progress if progress else "-"
            print(f"{exp_id:<10} {exp.name:<20} {status:<12} {progress_str:<20} {exp.description}")

        print("="*80)

    def run_daemon(self, interval: int = 30) -> None:
        """运行守护进程"""
        print("🛡️  实验管理器守护进程启动...")
        print(f"   检查间隔：{interval}秒")
        print(f"   按 Ctrl+C 停止\n")

        self.state["started_at"] = datetime.now().isoformat()
        self._save_state()

        try:
            while True:
                try:
                    self.show_status()
                    self.monitor_and_restart(max_restarts=3)
                    self.auto_schedule()

                    print(f"\n⏰ 下次检查：{datetime.now().strftime('%H:%M:%S')}")
                    print("─" * 80)

                    for _ in range(interval):
                        time.sleep(1)

                except KeyboardInterrupt:
                    break

        except KeyboardInterrupt:
            print("\n\n收到终止信号，正在清理...")

        finally:
            self.stop_all()
            self._save_state()
            print("✅ 实验管理器已停止")
