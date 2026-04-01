"""
统一入口 - Causal Feature Selection 实验框架

Usage:
    python main.py --config configs/baseline.yaml
    python main.py --config configs/exp1_attribution.yaml
    python main.py --config configs/exp4_causal.yaml
"""

import argparse
import yaml
import torch
import torch.nn as nn
from pathlib import Path
import json
from datetime import datetime
from typing import Dict, Any

from src.data.loader import IVRDataLoader
from src.models.baseline import get_baseline_model


def parse_args():
    parser = argparse.ArgumentParser(description="Causal Feature Selection for CTR")
    parser.add_argument("--config", type=str, required=True, help="Config file path")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    return parser.parse_args()


def load_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


class ExperimentRunner:
    """实验运行器"""
    
    def __init__(self, config: Dict[str, Any], device: str = "cuda"):
        self.config = config
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # 初始化组件
        self.data_loader = None
        self.model = None
        self.trainer = None
        
        # 结果存储
        self.results = {
            "config": config_path,
            "timestamp": datetime.now().isoformat(),
            "metrics": {}
        }
    
    def setup_data(self):
        """设置数据加载"""
        data_config = self.config.get("data", {})
        self.data_loader = IVRDataLoader(data_config.get("path"))
        
        # 加载数据
        days = data_config.get("days", 7)
        self.df = self.data_loader.load_ivr_sample(days=days)
        
        # 分析特征
        analysis = self.data_loader.analyze_features(self.df)
        print("\n=== Data Analysis ===")
        print(f"Total samples: {analysis['total_samples']:,}")
        print(f"Total features: {analysis['total_features']}")
        
        return analysis
    
    def setup_model(self, feature_config: Dict[str, int]):
        """设置模型"""
        model_config = self.config.get("model", {})
        model_type = model_config.get("type", "widedeep")
        
        self.model = get_baseline_model(
            model_type=model_type,
            feature_config=feature_config,
            **model_config.get("args", {})
        )
        
        self.model = self.model.to(self.device)
        num_params = sum(p.numel() for p in self.model.parameters())
        print(f"\n=== Model ===")
        print(f"Type: {model_type}")
        print(f"Parameters: {num_params:,}")
        
    def train(self):
        """训练流程"""
        train_config = self.config.get("training", {})
        
        # 优化器
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=train_config.get("lr", 5e-5)
        )
        
        # Loss
        criterion = nn.BCEWithLogitsLoss()
        
        # 训练循环（简化版）
        self.model.train()
        num_epochs = train_config.get("epochs", 1)
        
        for epoch in range(num_epochs):
            total_loss = 0
            num_batches = 0
            
            # TODO: 实现数据迭代器
            # for batch in dataloader:
            #     optimizer.zero_grad()
            #     outputs = self.model(batch["features"])
            #     loss = criterion(outputs, batch["labels"])
            #     loss.backward()
            #     optimizer.step()
            #     total_loss += loss.item()
            #     num_batches += 1
            
            print(f"Epoch {epoch+1}/{num_epochs} - Loss: {total_loss/num_batches:.4f}")
    
    def evaluate(self):
        """评估流程"""
        self.model.eval()
        
        # TODO: 实现评估逻辑
        metrics = {
            "auc": 0.0,
            "logloss": 0.0,
            "pcoc": 0.0
        }
        
        return metrics
    
    def save_results(self, output_dir: str = "results"):
        """保存结果"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # 生成唯一文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        exp_name = self.config.get("experiment", {}).get("name", "unnamed")
        filename = f"{exp_name}_{timestamp}.json"
        
        with open(output_path / filename, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        print(f"\nResults saved to: {output_path / filename}")


def main():
    args = parse_args()
    
    print("=" * 60)
    print("Causal Feature Selection for CTR Prediction")
    print("=" * 60)
    
    # 加载配置
    config = load_config(args.config)
    print(f"\nLoaded config: {args.config}")
    
    # 创建实验运行器
    runner = ExperimentRunner(config, device=args.device)
    
    # 执行实验
    try:
        # 1. 数据准备
        analysis = runner.setup_data()
        
        # 2. 特征工程（需要实际数据后才能确定 vocab_size）
        # 这里用占位符，后续完善
        feature_config = {
            "placeholder_feat": 1000  # 需要根据实际数据填充
        }
        
        # 3. 模型构建
        runner.setup_model(feature_config)
        
        # 4. 训练
        # runner.train()
        
        # 5. 评估
        # metrics = runner.evaluate()
        # runner.results["metrics"] = metrics
        
        # 6. 保存结果
        # runner.save_results()
        
        print("\n✅ Experiment framework initialized successfully!")
        print("⏳ Next steps: Implement data iterator and training loop")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
