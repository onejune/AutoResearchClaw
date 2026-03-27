"""
ChorusCVR 主实验脚本
"""
import os
import sys
import torch
import numpy as np
import random
from datetime import datetime

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import ExperimentConfig, DataConfig, ModelConfig, TrainConfig
from src.models.chorus_cvr import ChorusCVR, ESMM
from src.losses.chorus_loss import ChorusCVRLoss, ESMMLoss
from src.trainers.trainer import ChorusCVRTrainer
from data.dataloader import load_ali_ccp_data, create_dataloaders


def set_seed(seed: int = 42):
    """设置随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_experiment(config: ExperimentConfig, model_type: str = 'chorus'):
    """
    运行实验
    
    Args:
        config: 实验配置
        model_type: 'chorus' 或 'esmm'
    """
    print(f"\n{'='*60}")
    print(f"Running {model_type.upper()} Experiment")
    print(f"{'='*60}")
    
    # 设置随机种子
    set_seed(config.train.seed)
    
    # 设置设备
    device = config.train.device
    if device == 'cuda' and not torch.cuda.is_available():
        device = 'cpu'
        print("CUDA not available, using CPU")
    print(f"Using device: {device}")
    
    # 加载数据
    print("\n[1/4] Loading data...")
    train_df, test_df, label_encoders, sparse_feature_dims = load_ali_ccp_data(
        data_root=config.data.data_root,
        sparse_features=config.data.sparse_features,
        dense_features=config.data.dense_features,
        max_samples=config.data.max_samples,
        cache_dir='./data/cache'
    )
    
    # 过滤出存在的特征
    existing_sparse = [f for f in config.data.sparse_features if f in sparse_feature_dims]
    existing_dense = [f for f in config.data.dense_features if f in train_df.columns]
    
    # 创建 DataLoader
    train_loader, val_loader, test_loader = create_dataloaders(
        train_df=train_df,
        test_df=test_df,
        sparse_features=existing_sparse,
        dense_features=existing_dense,
        label_encoders=label_encoders,
        batch_size=config.train.batch_size,
        num_workers=4,
        val_ratio=config.data.val_ratio
    )
    
    print(f"Sparse features: {len(sparse_feature_dims)}")
    print(f"Dense features: {len(existing_dense)}")
    
    # 创建模型
    print("\n[2/4] Building model...")
    if model_type == 'chorus':
        model = ChorusCVR(
            sparse_feature_dims=sparse_feature_dims,
            dense_feature_num=len(existing_dense),
            embedding_dim=config.model.embedding_dim,
            shared_hidden_dims=config.model.shared_hidden_dims,
            tower_hidden_dims=config.model.tower_hidden_dims,
            dropout_rate=config.model.dropout_rate,
            activation=config.model.activation
        )
        loss_fn = ChorusCVRLoss(
            loss_weights=config.train.loss_weights,
            ipw_clip_min=config.train.ipw_clip_min,
            ipw_clip_max=config.train.ipw_clip_max
        )
    else:  # esmm
        model = ESMM(
            sparse_feature_dims=sparse_feature_dims,
            dense_feature_num=len(existing_dense),
            embedding_dim=config.model.embedding_dim,
            shared_hidden_dims=config.model.shared_hidden_dims,
            tower_hidden_dims=config.model.tower_hidden_dims,
            dropout_rate=config.model.dropout_rate,
            activation=config.model.activation
        )
        loss_fn = ESMMLoss()
    
    # 打印模型参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # 创建训练器
    trainer = ChorusCVRTrainer(
        model=model,
        loss_fn=loss_fn,
        learning_rate=config.train.learning_rate,
        weight_decay=config.train.weight_decay,
        device=device,
        log_interval=config.train.log_interval
    )
    
    # 训练
    print("\n[3/4] Training...")
    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=config.train.epochs,
        early_stop_patience=config.train.early_stop_patience
    )
    
    # 测试
    print("\n[4/4] Testing...")
    test_metrics = trainer.evaluate(test_loader, prefix='test')
    
    print(f"\n{'='*50}")
    print("Test Results:")
    print(f"{'='*50}")
    print(f"CTR-AUC:   {test_metrics['test_ctr_auc']:.4f}")
    print(f"CVR-AUC:   {test_metrics['test_cvr_auc']:.4f}")
    print(f"CTCVR-AUC: {test_metrics['test_ctcvr_auc']:.4f}")
    print(f"PCOC:      {test_metrics['test_pcoc']:.4f}")
    
    # 保存结果
    os.makedirs(config.output_dir, exist_ok=True)
    result_file = os.path.join(config.output_dir, f'{model_type}_results.txt')
    with open(result_file, 'w') as f:
        f.write(f"Experiment: {model_type.upper()}\n")
        f.write(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"\nTest Results:\n")
        for k, v in test_metrics.items():
            f.write(f"  {k}: {v:.4f}\n")
    
    print(f"\nResults saved to {result_file}")
    
    # 保存模型
    model_path = os.path.join(config.output_dir, f'{model_type}_model.pt')
    trainer.save_model(model_path)
    
    return test_metrics


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='ChorusCVR Experiment')
    parser.add_argument('--model', type=str, default='chorus', choices=['chorus', 'esmm'],
                        help='Model type: chorus or esmm')
    parser.add_argument('--max_samples', type=int, default=None,
                        help='Maximum number of samples (for quick testing)')
    parser.add_argument('--batch_size', type=int, default=4096,
                        help='Batch size')
    parser.add_argument('--epochs', type=int, default=1,
                        help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device: cuda or cpu')
    args = parser.parse_args()
    
    # 创建配置
    config = ExperimentConfig(
        data=DataConfig(max_samples=args.max_samples),
        model=ModelConfig(),
        train=TrainConfig(
            batch_size=args.batch_size,
            epochs=args.epochs,
            learning_rate=args.lr,
            device=args.device
        ),
        exp_name=f'{args.model}_exp',
        output_dir='./results'
    )
    
    # 运行实验
    run_experiment(config, model_type=args.model)


if __name__ == '__main__':
    main()
