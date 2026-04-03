#!/usr/bin/env python3
"""
实验配置管理器 - 动态管理实验队列
支持从配置文件加载/更新实验列表
"""

import json
import yaml
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
from datetime import datetime

@dataclass
class ExperimentConfig:
    """实验配置数据类"""
    name: str
    script: str
    result_dir: str
    priority: int
    description: str
    completed: bool = False
    max_restarts: int = 3
    use_gpu: bool = True
    min_memory_mb: int = 4096
    tags: List[str] = None
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = []

class ExperimentQueueManager:
    """实验队列管理器 - 支持动态增删改查"""

    def __init__(self, project_root: Path = None):
        # 自动推断项目根目录：本脚本在 scripts/ 下，上一级即项目根
        if project_root is None:
            project_root = Path(__file__).resolve().parent.parent
        self.project_root = project_root

        # 读取集中配置
        self._raw_conf = self._parse_conf(project_root / "expkit.conf")
        self.CONFIG_FILE = project_root / self._raw_conf.get("EXPERIMENTS_YAML", "experiments_config.yaml")
        self.JSON_FILE = project_root / self._raw_conf.get("EXPERIMENTS_JSON", "experiments_config.json")
        self.BACKUP_DIR = project_root / "config_backups"

        self.BACKUP_DIR.mkdir(parents=True, exist_ok=True)
        self.experiments: Dict[str, ExperimentConfig] = {}
        self._load_config()

    @staticmethod
    def _parse_conf(conf_file: Path) -> dict:
        """解析 expkit.conf"""
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
    
    def _load_config(self):
        """加载配置文件"""
        if not self.CONFIG_FILE.exists():
            self._create_default_config()
        
        with open(self.CONFIG_FILE, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        for exp_id, exp_data in config.get('experiments', {}).items():
            self.experiments[exp_id] = ExperimentConfig(**exp_data)
    
    def _create_default_config(self):
        """创建默认配置文件"""
        default_config = {
            "version": "1.0",
            "created_at": datetime.now().isoformat(),
            "experiments": {
                "exp001": {
                    "name": "AutoEmb",
                    "script": "scripts/run_exp001_autoemb_v3.py",
                    "result_dir": "exp001_autoemb",
                    "priority": 1,
                    "description": "自动化维度分配",
                    "completed": False,
                    "max_restarts": 3,
                    "use_gpu": True,
                    "tags": ["embedding", "auto"]
                },
                "exp002": {
                    "name": "DDS",
                    "script": "scripts/run_exp002_dds.py",
                    "result_dir": "exp002_dds",
                    "priority": 2,
                    "description": "数据分布搜索",
                    "completed": False,
                    "max_restarts": 3,
                    "use_gpu": True,
                    "tags": ["distribution", "search"]
                },
                "exp003": {
                    "name": "Hierarchical v2",
                    "script": "scripts/run_exp003_hierarchical_v2.py",
                    "result_dir": "exp003_hierarchical_v2",
                    "priority": 3,
                    "description": "层次化 Embedding (deep chain)",
                    "completed": True,
                    "max_restarts": 3,
                    "use_gpu": True,
                    "tags": ["hierarchical", "embedding"]
                },
                "exp004": {
                    "name": "MetaEmb",
                    "script": "scripts/run_exp004_metaemb.py",
                    "result_dir": "exp004_metaemb",
                    "priority": 4,
                    "description": "元学习冷启动",
                    "completed": False,
                    "max_restarts": 3,
                    "use_gpu": False,
                    "tags": ["meta-learning", "cold-start"]
                },
                "exp005": {
                    "name": "Contrastive",
                    "script": "scripts/run_exp005_contrastive.py",
                    "result_dir": "exp005_contrastive",
                    "priority": 5,
                    "description": "对比学习增强",
                    "completed": False,
                    "max_restarts": 3,
                    "use_gpu": False,
                    "tags": ["contrastive", "representation"]
                },
                "exp006": {
                    "name": "FiBiNET",
                    "script": "scripts/run_exp006_fibinet.py",
                    "result_dir": "exp006_fibinet",
                    "priority": 6,
                    "description": "特征交互",
                    "completed": False,
                    "max_restarts": 3,
                    "use_gpu": False,
                    "tags": ["fibinet", "interaction"]
                },
                "exp007": {
                    "name": "Combined",
                    "script": "scripts/run_exp007_combined.py",
                    "result_dir": "exp007_combined",
                    "priority": 7,
                    "description": "综合方案",
                    "completed": False,
                    "max_restarts": 3,
                    "use_gpu": False,
                    "tags": ["combined", "ensemble"]
                }
            }
        }
        
        with open(self.CONFIG_FILE, 'w', encoding='utf-8') as f:
            yaml.dump(default_config, f, allow_unicode=True, default_flow_style=False)
    
    def _save_config(self):
        """保存配置文件"""
        config = {
            "version": "1.0",
            "updated_at": datetime.now().isoformat(),
            "experiments": {}
        }
        
        for exp_id, exp_config in self.experiments.items():
            config["experiments"][exp_id] = asdict(exp_config)
        
        with open(self.CONFIG_FILE, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, allow_unicode=True, default_flow_style=False)
    
    def _backup_config(self):
        """备份配置文件"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_file = self.BACKUP_DIR / f"config_backup_{timestamp}.yaml"
        
        with open(self.CONFIG_FILE, 'r', encoding='utf-8') as f:
            content = f.read()
        
        with open(backup_file, 'w', encoding='utf-8') as f:
            f.write(content)
        
        return backup_file
    
    def add_experiment(self, exp_id: str, config: Dict) -> bool:
        """添加新实验"""
        if exp_id in self.experiments:
            print(f"⚠️  实验 {exp_id} 已存在")
            return False
        
        # 验证必填字段
        required_fields = ['name', 'script', 'result_dir', 'priority', 'description']
        for field in required_fields:
            if field not in config:
                print(f"❌ 缺少必填字段：{field}")
                return False
        
        # 备份配置
        self._backup_config()
        
        # 添加实验
        self.experiments[exp_id] = ExperimentConfig(**config)
        self._save_config()
        
        print(f"✅ 已添加实验：{exp_id} - {config['name']}")
        return True
    
    def update_experiment(self, exp_id: str, **kwargs) -> bool:
        """更新实验配置"""
        if exp_id not in self.experiments:
            print(f"❌ 实验不存在：{exp_id}")
            return False
        
        # 备份配置
        self._backup_config()
        
        # 更新字段
        exp = self.experiments[exp_id]
        for key, value in kwargs.items():
            if hasattr(exp, key):
                setattr(exp, key, value)
            else:
                print(f"⚠️  未知字段：{key}")
        
        self._save_config()
        print(f"✅ 已更新实验：{exp_id}")
        return True
    
    def remove_experiment(self, exp_id: str, keep_results: bool = True) -> bool:
        """移除实验"""
        if exp_id not in self.experiments:
            print(f"❌ 实验不存在：{exp_id}")
            return False
        
        # 备份配置
        self._backup_config()
        
        # 删除实验
        del self.experiments[exp_id]
        self._save_config()
        
        print(f"✅ 已移除实验：{exp_id}")
        return True
    
    def mark_completed(self, exp_id: str):
        """标记实验为已完成"""
        if exp_id in self.experiments:
            self.update_experiment(exp_id, completed=True)
            print(f"✅ 已标记完成：{exp_id}")
    
    def get_pending_experiments(self) -> List[tuple]:
        """获取待运行的实验（按优先级排序）"""
        pending = []
        for exp_id, exp_config in self.experiments.items():
            if not exp_config.completed:
                pending.append((exp_id, exp_config.priority, exp_config))
        
        pending.sort(key=lambda x: x[1])
        return pending
    
    def list_experiments(self, status: str = None):
        """列出所有实验"""
        print("\n" + "="*100)
        print("📋 实验队列")
        print("="*100)
        print(f"{'ID':<10} {'名称':<20} {'优先级':<8} {'状态':<12} {'GPU':<6} {'描述'}")
        print("-"*100)
        
        for exp_id in sorted(self.experiments.keys()):
            exp = self.experiments[exp_id]
            status_str = "✅ 已完成" if exp.completed else "⏸️  待运行"
            gpu_str = "✅" if exp.use_gpu else "❌"
            
            if status is None or (status == "pending" and not exp.completed) or \
               (status == "completed" and exp.completed):
                print(f"{exp_id:<10} {exp.name:<20} {exp.priority:<8} {status_str:<12} {gpu_str:<6} {exp.description}")
        
        print("="*100)
    
    def export_to_json(self, output_file: str = None):
        """导出为 JSON 格式（供 experiment_manager.py 使用）"""
        data = {}
        for exp_id, exp_config in self.experiments.items():
            data[exp_id] = asdict(exp_config)

        target = Path(output_file) if output_file else self.JSON_FILE

        with open(target, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        print(f"✅ 已导出到：{target}")


def main():
    """命令行接口"""
    import argparse
    
    parser = argparse.ArgumentParser(description="实验队列管理器")
    subparsers = parser.add_subparsers(dest='command', help='命令')
    
    # list 命令
    list_parser = subparsers.add_parser('list', help='列出实验')
    list_parser.add_argument('--status', choices=['all', 'pending', 'completed'], default='all')
    
    # add 命令
    add_parser = subparsers.add_parser('add', help='添加实验')
    add_parser.add_argument('--id', required=True, help='实验 ID')
    add_parser.add_argument('--name', required=True, help='实验名称')
    add_parser.add_argument('--script', required=True, help='脚本路径')
    add_parser.add_argument('--result-dir', required=True, help='结果目录')
    add_parser.add_argument('--priority', type=int, required=True, help='优先级')
    add_parser.add_argument('--description', required=True, help='描述')
    add_parser.add_argument('--use-gpu', action='store_true', default=True, help='使用 GPU')
    add_parser.add_argument('--no-gpu', action='store_false', dest='use_gpu', help='不使用 GPU')
    
    # update 命令
    update_parser = subparsers.add_parser('update', help='更新实验')
    update_parser.add_argument('--id', required=True, help='实验 ID')
    update_parser.add_argument('--priority', type=int, help='新优先级')
    update_parser.add_argument('--completed', action='store_true', help='标记为完成')
    update_parser.add_argument('--not-completed', action='store_true', dest='not_completed', help='标记为未完成')
    
    # remove 命令
    remove_parser = subparsers.add_parser('remove', help='移除实验')
    remove_parser.add_argument('--id', required=True, help='实验 ID')
    
    # export 命令
    subparsers.add_parser('export', help='导出为 JSON')
    
    args = parser.parse_args()
    
    manager = ExperimentQueueManager()
    
    if args.command == 'list':
        manager.list_experiments(status=args.status if args.status != 'all' else None)
    
    elif args.command == 'add':
        config = {
            'name': args.name,
            'script': args.script,
            'result_dir': args.result_dir,
            'priority': args.priority,
            'description': args.description,
            'use_gpu': args.use_gpu,
            'completed': False,
            'max_restarts': 3,
            'min_memory_mb': 4096,
            'tags': []
        }
        manager.add_experiment(args.id, config)
    
    elif args.command == 'update':
        updates = {}
        if args.priority is not None:
            updates['priority'] = args.priority
        if args.completed:
            updates['completed'] = True
        if args.not_completed:
            updates['completed'] = False
        
        if updates:
            manager.update_experiment(args.id, **updates)
        else:
            print("⚠️  请指定至少一个要更新的字段")
    
    elif args.command == 'remove':
        manager.remove_experiment(args.id)
    
    elif args.command == 'export':
        manager.export_to_json()
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
