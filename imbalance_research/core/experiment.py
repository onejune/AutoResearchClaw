"""
实验类 - 封装完整的训练流程

特点:
1. 自动保存 checkpoint 和结果
2. 支持断点续训
3. 完整的日志记录
4. 结果不会丢失
"""

import os
import sys
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional
from tqdm import tqdm
import hashlib

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from models.losses import create_focal_loss, DynamicFocalLoss
from models.embedding_model import create_model_with_embeddings
from data.dataset import load_ivr_dataset
from evaluation.metrics import compute_metrics


class Experiment:
    """实验类 - 管理完整的训练生命周期"""
    
    def __init__(self, name: str, config: Dict[str, Any]):
        self.name = name
        self.config = config
        self.start_time = datetime.now()
        
        # 生成唯一实验 ID (基于配置哈希)
        config_str = json.dumps(config, sort_keys=True)
        self.exp_id = hashlib.md5(config_str.encode()).hexdigest()[:8]
        
        # 目录设置
        self.exp_dir = PROJECT_ROOT / 'results' / 'experiments' / f"{name}_{self.exp_id}"
        self.exp_dir.mkdir(parents=True, exist_ok=True)
        
        self.checkpoint_dir = self.exp_dir / 'checkpoints'
        self.checkpoint_dir.mkdir(exist_ok=True)
        
        # 保存配置
        with open(self.exp_dir / 'config.json', 'w') as f:
            json.dump(config, f, indent=2, default=str)
        
        # 初始化组件
        self.device = self._setup_device()
        self.model = None
        self.loss_fn = None
        self.optimizer = None
        self.train_loader = None
        self.val_loader = None
        
        # 训练状态
        self.current_epoch = 0
        self.best_val_auc = 0.0
        self.train_history = []
        self.val_history = []
        
        print(f"\n🔬 实验初始化: {name}")
        print(f"   ID: {self.exp_id}")
        print(f"   目录: {self.exp_dir}")
    
    def _setup_device(self) -> torch.device:
        """设置计算设备"""
        gpu_config = self.config.get('gpu', {})
        device_id = gpu_config.get('device', -1)
        
        if device_id >= 0:
            device = torch.device(f'cuda:{device_id}')
        elif torch.cuda.is_available():
            # 自动选择显存最空闲的 GPU
            device = self._auto_select_gpu()
        else:
            device = torch.device('cpu')
        
        print(f"   设备: {device}")
        return device
    
    def _auto_select_gpu(self) -> torch.device:
        """自动选择最空闲的 GPU (支持多进程竞争)"""
        try:
            import subprocess
            import os
            import fcntl
            import time
            
            # 获取 GPU 状态
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=memory.free,utilization.gpu', '--format=csv,nounits,noheader'],
                capture_output=True, text=True
            )
            lines = result.stdout.strip().split('\n')
            gpu_stats = []
            for i, line in enumerate(lines):
                parts = line.split(',')
                free_mem = int(parts[0].strip())
                util = int(parts[1].strip())
                gpu_stats.append((i, free_mem, util))
            
            # 使用文件锁避免多进程同时选择同一 GPU
            lock_dir = '/tmp/gpu_locks'
            os.makedirs(lock_dir, exist_ok=True)
            
            # 按空闲显存排序，优先选显存最多的
            gpu_stats.sort(key=lambda x: (-x[1], x[2]))
            
            for gpu_id, free_mem, util in gpu_stats:
                lock_file = f'{lock_dir}/gpu_{gpu_id}.lock'
                try:
                    fd = os.open(lock_file, os.O_CREAT | os.O_RDWR)
                    # 非阻塞尝试获取锁
                    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    # 获取成功，记录 PID
                    os.write(fd, f'{os.getpid()}\n'.encode())
                    print(f"   自动选择 GPU {gpu_id} (空闲: {free_mem} MB, 利用率: {util}%)")
                    # 保存 fd 以便后续释放
                    self._gpu_lock_fd = fd
                    return torch.device(f'cuda:{gpu_id}')
                except (IOError, OSError):
                    # 锁被占用，尝试下一个
                    try:
                        os.close(fd)
                    except:
                        pass
                    continue
            
            # 所有 GPU 都被锁定，选显存最多的（共享使用）
            best_gpu = gpu_stats[0][0]
            print(f"   所有 GPU 繁忙，共享 GPU {best_gpu} (空闲: {gpu_stats[0][1]} MB)")
            return torch.device(f'cuda:{best_gpu}')
            
        except Exception as e:
            print(f"   GPU 自动选择失败: {e}，使用 GPU 0")
            return torch.device('cuda:0')
    
    def _setup_data(self):
        """加载数据"""
        data_config = self.config.get('data', {})
        sample_ratio = data_config.get('sample_ratio', 1.0)
        
        print("\n📂 加载数据...")
        if sample_ratio < 1.0:
            print(f"   采样比例: {sample_ratio*100:.1f}%")
        
        exclude_features = data_config.get('exclude_features', [])
        
        wrapper = load_ivr_dataset(
            data_dir=data_config.get('dir'),
            label_col=data_config.get('label_col', 'ctcvr_label'),
            sample_ratio=sample_ratio,
            exclude_features=exclude_features,
        )
        
        train_config = self.config.get('training', {})
        self.train_loader, self.val_loader, self.test_loader = wrapper.get_dataloaders(
            batch_size=train_config.get('batch_size', 256),
            num_workers=train_config.get('num_workers', 4),
        )
        
        # 获取特征维度 (处理 Subset 嵌套)
        actual_dataset = wrapper.train_dataset
        while hasattr(actual_dataset, 'dataset'):
            actual_dataset = actual_dataset.dataset
        
        self.feature_cols = actual_dataset.feature_cols
        self.feature_dims = []
        
        df_features = actual_dataset.df[self.feature_cols]
        for col in self.feature_cols:
            max_val = df_features[col].max() + 1
            self.feature_dims.append(int(max_val))
        
        print(f"   特征数: {len(self.feature_cols)}")
        print(f"   训练批次: {len(self.train_loader)}")
        print(f"   验证批次: {len(self.val_loader)}")
    
    def _setup_model(self):
        """创建模型"""
        model_config = self.config.get('model', {})
        
        print("\n🏗️  创建模型...")
        self.model = create_model_with_embeddings(
            model_type=model_config.get('type', 'mlp'),
            feature_dims=self.feature_dims,
            hidden_dims=model_config.get('hidden_dims', [256, 128, 64]),
            dropout=model_config.get('dropout', 0.3),
            embed_dim=model_config.get('embed_dim', 16),
        )
        self.model = self.model.to(self.device)
        
        params_mb = sum(p.numel() * 4 for p in self.model.parameters()) / (1024**2)
        print(f"   类型: {model_config.get('type', 'mlp')}")
        print(f"   参数量: {params_mb:.2f} MB")
    
    def _setup_loss(self):
        """创建损失函数"""
        loss_config = self.config.get('loss', {})
        loss_type = loss_config.get('type', 'focal')
        
        print(f"\n⚡ 创建 Loss: {loss_type}")
        
        # 提取 loss 参数
        loss_kwargs = {k: v for k, v in loss_config.items() if k != 'type'}
        
        # Dynamic Focal Loss 需要 total_epochs
        if loss_type == 'dynamic':
            train_config = self.config.get('training', {})
            loss_kwargs['total_epochs'] = train_config.get('epochs', 20)
        
        self.loss_fn = create_focal_loss(loss_type=loss_type, **loss_kwargs)
        print(f"   {self.loss_fn.__class__.__name__}")
    
    def _setup_optimizer(self):
        """创建优化器"""
        train_config = self.config.get('training', {})
        lr = train_config.get('learning_rate', 1e-3)
        
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        print(f"\n🔧 优化器: Adam (lr={lr})")
    
    def _train_epoch(self) -> float:
        """训练一个 epoch"""
        self.model.train()
        total_loss = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch+1}", leave=False)
        for batch_x, batch_y in pbar:
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(batch_x)
            loss = self.loss_fn(outputs, batch_y)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        return total_loss / len(self.train_loader)
    
    @torch.no_grad()
    def _validate(self) -> tuple:
        """验证"""
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_targets = []
        
        for batch_x, batch_y in self.val_loader:
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)
            
            outputs = self.model(batch_x)
            loss = self.loss_fn(outputs, batch_y)
            
            total_loss += loss.item()
            # 转换为概率
            probs = torch.sigmoid(outputs).cpu().numpy()
            all_preds.extend(probs)
            all_targets.extend(batch_y.cpu().numpy())
        
        avg_loss = total_loss / len(self.val_loader)
        metrics = compute_metrics(all_preds, all_targets)
        
        return avg_loss, metrics
    
    def _save_checkpoint(self, is_best: bool = False):
        """保存 checkpoint"""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_auc': self.best_val_auc,
            'train_history': self.train_history,
            'val_history': self.val_history,
            'config': self.config,
        }
        
        # 保存最新
        torch.save(checkpoint, self.checkpoint_dir / 'latest.pt')
        
        # 保存最优
        if is_best:
            torch.save(checkpoint, self.checkpoint_dir / 'best.pt')
    
    def load_checkpoint(self, path: str):
        """加载 checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.current_epoch = checkpoint['epoch']
        self.best_val_auc = checkpoint['best_val_auc']
        self.train_history = checkpoint['train_history']
        self.val_history = checkpoint['val_history']
        
        if self.model is not None:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        if self.optimizer is not None:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        print(f"📂 已加载 checkpoint: epoch {self.current_epoch}, best_auc {self.best_val_auc:.4f}")
    
    def _save_results(self):
        """保存最终结果"""
        results = {
            'experiment': self.name,
            'exp_id': self.exp_id,
            'config': self.config,
            'best_val_auc': self.best_val_auc,
            'total_epochs': self.current_epoch + 1,
            'train_history': self.train_history,
            'val_history': self.val_history,
            'start_time': self.start_time.isoformat(),
            'end_time': datetime.now().isoformat(),
        }
        
        with open(self.exp_dir / 'results.json', 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\n💾 结果已保存: {self.exp_dir / 'results.json'}")
    
    def run(self) -> Dict[str, Any]:
        """运行完整实验"""
        print("\n" + "=" * 60)
        print(f"🚀 开始实验: {self.name}")
        print("=" * 60)
        
        # 初始化
        self._setup_data()
        self._setup_model()
        self._setup_loss()
        self._setup_optimizer()
        
        train_config = self.config.get('training', {})
        num_epochs = train_config.get('epochs', 20)
        patience = train_config.get('patience', 5)
        patience_counter = 0
        
        print(f"\n📈 开始训练 ({num_epochs} epochs, patience={patience})")
        print("-" * 60)
        
        for epoch in range(self.current_epoch, num_epochs):
            self.current_epoch = epoch
            
            # Dynamic Focal Loss 更新
            if isinstance(self.loss_fn, DynamicFocalLoss):
                self.loss_fn.set_epoch(epoch)
            
            # 训练
            train_loss = self._train_epoch()
            
            # 验证
            val_loss, val_metrics = self._validate()
            
            # 记录
            self.train_history.append({'epoch': epoch, 'loss': train_loss})
            self.val_history.append({'epoch': epoch, 'loss': val_loss, **val_metrics})
            
            # 打印
            gamma_str = f" | γ={self.loss_fn.gamma:.2f}" if hasattr(self.loss_fn, 'gamma') else ""
            print(f"Epoch {epoch+1:3d}/{num_epochs} | "
                  f"Train: {train_loss:.4f} | "
                  f"Val: {val_loss:.4f} | "
                  f"AUC: {val_metrics['auc']:.4f} | "
                  f"PCOC: {val_metrics['pcoc']:.4f}{gamma_str}")
            
            # 保存最优
            is_best = val_metrics['auc'] > self.best_val_auc
            if is_best:
                self.best_val_auc = val_metrics['auc']
                patience_counter = 0
                print(f"         ⭐ 新最优 AUC: {self.best_val_auc:.4f}")
            else:
                patience_counter += 1
            
            # 保存 checkpoint
            self._save_checkpoint(is_best=is_best)
            
            # 早停
            if patience_counter >= patience:
                print(f"\n⚠️  早停触发 (patience={patience})")
                break
        
        # 保存结果
        self._save_results()
        
        print("\n" + "=" * 60)
        print(f"✅ 实验完成: {self.name}")
        print(f"   最佳 Val AUC: {self.best_val_auc:.4f}")
        print(f"   结果目录: {self.exp_dir}")
        print("=" * 60)
        
        return {
            'experiment': self.name,
            'best_val_auc': self.best_val_auc,
            'total_epochs': self.current_epoch + 1,
            'exp_dir': str(self.exp_dir),
        }
