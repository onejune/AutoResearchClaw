"""
完整基线对比实验
包含: ESMM, ESCM2-IPW, ESCM2-DR, DCMT, DDPO, ChorusCVR
"""
import os
import sys
import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
from datetime import datetime
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.chorus_cvr import ChorusCVR
from src.models.baselines import ESMM, ESCM2, DCMT, DDPO
from src.losses.chorus_loss import ChorusCVRLoss
from src.losses.baseline_losses import ESMMLoss, ESCM2Loss, DCMTLoss, DDPOLoss


def load_data(max_samples=None):
    """加载数据"""
    train_path = '/mnt/data/oss_wanjun/pai_work/open_research/dataset/ali_ccp/ali_ccp_train.csv'
    test_path = '/mnt/data/oss_wanjun/pai_work/open_research/dataset/ali_ccp/ali_ccp_test.csv'
    
    print(f"Loading data (max_samples={max_samples})...")
    train_df = pd.read_csv(train_path, nrows=max_samples)
    test_df = pd.read_csv(test_path, nrows=max_samples // 5 if max_samples else None)
    
    sparse_features = ['101', '121', '122', '124', '125', '126', '127', '128', '129',
                       '205', '206', '207', '210', '216', '508', '509', '702', '853', '301']
    dense_features = ['D109_14', 'D110_14', 'D127_14', 'D150_14', 'D508', 'D509', 'D702', 'D853']
    
    # 编码稀疏特征
    label_encoders = {}
    sparse_feature_dims = {}
    
    for feat in sparse_features:
        le = LabelEncoder()
        all_values = pd.concat([train_df[feat].astype(str), test_df[feat].astype(str)])
        le.fit(all_values)
        train_df[feat] = le.transform(train_df[feat].astype(str))
        test_df[feat] = le.transform(test_df[feat].astype(str))
        label_encoders[feat] = le
        sparse_feature_dims[feat] = len(le.classes_) + 1
    
    train_df[dense_features] = train_df[dense_features].fillna(0)
    test_df[dense_features] = test_df[dense_features].fillna(0)
    
    print(f"Train: {len(train_df)}, Test: {len(test_df)}")
    print(f"Click rate: {train_df['click'].mean():.4f}")
    print(f"CVR (click space): {train_df[train_df['click']==1]['purchase'].mean():.4f}")
    
    return train_df, test_df, sparse_features, dense_features, sparse_feature_dims


def create_batch(df, sparse_features, dense_features, device='cpu'):
    """创建批次数据"""
    sparse_tensors = {
        feat: torch.tensor(df[feat].values, dtype=torch.long, device=device)
        for feat in sparse_features
    }
    dense_tensor = torch.tensor(df[dense_features].values, dtype=torch.float32, device=device)
    click_labels = torch.tensor(df['click'].values, dtype=torch.float32, device=device)
    conversion_labels = torch.tensor(df['purchase'].values, dtype=torch.float32, device=device)
    
    return sparse_tensors, dense_tensor, click_labels, conversion_labels


def train_model(model, loss_fn, train_df, sparse_features, dense_features, 
                batch_size=4096, epochs=1, lr=1e-3, device='cpu', model_name='model'):
    """训练模型"""
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    
    n_samples = len(train_df)
    n_batches = (n_samples + batch_size - 1) // batch_size
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        train_df_shuffled = train_df.sample(frac=1).reset_index(drop=True)
        
        pbar = tqdm(range(n_batches), desc=f'{model_name} Epoch {epoch+1}')
        for i in pbar:
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, n_samples)
            batch_df = train_df_shuffled.iloc[start_idx:end_idx]
            
            sparse_tensors, dense_tensor, click_labels, conversion_labels = create_batch(
                batch_df, sparse_features, dense_features, device
            )
            
            optimizer.zero_grad()
            outputs = model(sparse_tensors, dense_tensor)
            loss, loss_dict = loss_fn(outputs, click_labels, conversion_labels)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss_dict['total']
            
            if (i + 1) % 100 == 0:
                pbar.set_postfix({'loss': f"{total_loss / (i+1):.4f}"})
        
        print(f"{model_name} Epoch {epoch+1} - Avg Loss: {total_loss / n_batches:.4f}")
    
    return model


@torch.no_grad()
def evaluate_model(model, test_df, sparse_features, dense_features, device='cpu', batch_size=8192):
    """评估模型"""
    model.eval()
    
    all_pCTR = []
    all_pCVR = []
    all_pCTCVR = []
    all_click_labels = []
    all_conversion_labels = []
    
    n_samples = len(test_df)
    n_batches = (n_samples + batch_size - 1) // batch_size
    
    for i in range(n_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, n_samples)
        batch_df = test_df.iloc[start_idx:end_idx]
        
        sparse_tensors, dense_tensor, click_labels, conversion_labels = create_batch(
            batch_df, sparse_features, dense_features, device
        )
        
        outputs = model(sparse_tensors, dense_tensor)
        
        all_pCTR.append(outputs['pCTR'].cpu().numpy())
        all_pCVR.append(outputs['pCVR'].cpu().numpy())
        all_pCTCVR.append(outputs['pCTCVR'].cpu().numpy())
        all_click_labels.append(click_labels.cpu().numpy())
        all_conversion_labels.append(conversion_labels.cpu().numpy())
    
    all_pCTR = np.concatenate(all_pCTR)
    all_pCVR = np.concatenate(all_pCVR)
    all_pCTCVR = np.concatenate(all_pCTCVR)
    all_click_labels = np.concatenate(all_click_labels)
    all_conversion_labels = np.concatenate(all_conversion_labels)
    
    results = {}
    
    # CTR AUC
    results['ctr_auc'] = roc_auc_score(all_click_labels, all_pCTR)
    
    # CVR AUC (点击空间)
    click_mask = all_click_labels > 0.5
    if click_mask.sum() > 0 and all_conversion_labels[click_mask].sum() > 0:
        results['cvr_auc'] = roc_auc_score(all_conversion_labels[click_mask], all_pCVR[click_mask])
    else:
        results['cvr_auc'] = 0.5
    
    # CTCVR AUC
    ctcvr_labels = all_click_labels * all_conversion_labels
    if ctcvr_labels.sum() > 0:
        results['ctcvr_auc'] = roc_auc_score(ctcvr_labels, all_pCTCVR)
    else:
        results['ctcvr_auc'] = 0.5
    
    # PCOC
    if click_mask.sum() > 0:
        actual_cvr = all_conversion_labels[click_mask].mean()
        pred_cvr = all_pCVR[click_mask].mean()
        results['pcoc'] = pred_cvr / actual_cvr if actual_cvr > 0 else 1.0
    else:
        results['pcoc'] = 1.0
    
    return results


def create_model(model_type, sparse_feature_dims, dense_feature_num, config):
    """创建模型和损失函数"""
    common_args = {
        'sparse_feature_dims': sparse_feature_dims,
        'dense_feature_num': dense_feature_num,
        'embedding_dim': config['embedding_dim'],
        'shared_hidden_dims': config['shared_hidden_dims'],
        'tower_hidden_dims': config['tower_hidden_dims'],
        'dropout_rate': config['dropout_rate'],
    }
    
    if model_type == 'esmm':
        model = ESMM(**common_args)
        loss_fn = ESMMLoss()
    elif model_type == 'escm2_ipw':
        model = ESCM2(**common_args, mode='ipw')
        loss_fn = ESCM2Loss(mode='ipw')
    elif model_type == 'escm2_dr':
        model = ESCM2(**common_args, mode='dr')
        loss_fn = ESCM2Loss(mode='dr')
    elif model_type == 'dcmt':
        model = DCMT(**common_args)
        loss_fn = DCMTLoss()
    elif model_type == 'ddpo':
        model = DDPO(**common_args)
        loss_fn = DDPOLoss()
    elif model_type == 'chorus':
        model = ChorusCVR(**common_args)
        loss_fn = ChorusCVRLoss(
            loss_weights={
                'ctcvr': 1.0,
                'cvr_ipw': 1.0,
                'ctuncvr': 1.0,
                'uncvr_ipw': 1.0,
                'align_ipw': 0.5,
            }
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return model, loss_fn


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_samples', type=int, default=5000000, help='Max training samples')
    parser.add_argument('--batch_size', type=int, default=4096, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=1, help='Number of epochs')
    parser.add_argument('--device', type=str, default='cpu', help='Device')
    args = parser.parse_args()
    
    print("="*70)
    print("ChorusCVR Full Baseline Comparison")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Max samples: {args.max_samples}")
    print("="*70)
    
    # 加载数据
    train_df, test_df, sparse_features, dense_features, sparse_feature_dims = load_data(args.max_samples)
    
    # 模型配置
    config = {
        'embedding_dim': 16,
        'shared_hidden_dims': [256, 128],
        'tower_hidden_dims': [64, 32],
        'dropout_rate': 0.1,
    }
    
    # 所有要对比的模型
    model_types = ['esmm', 'escm2_ipw', 'dcmt', 'ddpo', 'chorus']
    
    all_results = {}
    
    for model_type in model_types:
        print(f"\n{'='*70}")
        print(f"Training {model_type.upper()}")
        print("="*70)
        
        model, loss_fn = create_model(
            model_type, sparse_feature_dims, len(dense_features), config
        )
        
        # 打印参数量
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Parameters: {total_params:,}")
        
        # 训练
        model = train_model(
            model, loss_fn, train_df, sparse_features, dense_features,
            batch_size=args.batch_size, epochs=args.epochs, lr=args.lr,
            device=args.device, model_name=model_type.upper()
        )
        
        # 评估
        results = evaluate_model(model, test_df, sparse_features, dense_features, args.device)
        all_results[model_type] = results
        
        print(f"\n{model_type.upper()} Results:")
        for k, v in results.items():
            print(f"  {k}: {v:.4f}")
    
    # 打印对比结果
    print("\n" + "="*70)
    print("Final Comparison (Test Set)")
    print("="*70)
    
    print(f"\n{'Model':<15} {'CTR-AUC':<12} {'CVR-AUC':<12} {'CTCVR-AUC':<12} {'PCOC':<12}")
    print("-"*63)
    for model_type in model_types:
        r = all_results[model_type]
        print(f"{model_type.upper():<15} {r['ctr_auc']:<12.4f} {r['cvr_auc']:<12.4f} {r['ctcvr_auc']:<12.4f} {r['pcoc']:<12.4f}")
    
    # 计算相对 ESMM 的提升
    esmm_results = all_results['esmm']
    print(f"\n{'Model':<15} {'CVR Δ':<12} {'CTCVR Δ':<12}")
    print("-"*39)
    for model_type in model_types:
        if model_type == 'esmm':
            continue
        r = all_results[model_type]
        cvr_diff = r['cvr_auc'] - esmm_results['cvr_auc']
        ctcvr_diff = r['ctcvr_auc'] - esmm_results['ctcvr_auc']
        print(f"{model_type.upper():<15} {cvr_diff:+.4f}      {ctcvr_diff:+.4f}")
    
    # 保存结果
    os.makedirs('./results', exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    with open(f'./results/baseline_comparison_{timestamp}.md', 'w') as f:
        f.write(f"# ChorusCVR Baseline Comparison\n\n")
        f.write(f"**Time**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"## Configuration\n")
        f.write(f"- Train samples: {len(train_df)}\n")
        f.write(f"- Test samples: {len(test_df)}\n")
        f.write(f"- Batch size: {args.batch_size}\n")
        f.write(f"- Learning rate: {args.lr}\n")
        f.write(f"- Epochs: {args.epochs}\n\n")
        
        f.write(f"## Results\n\n")
        f.write(f"| Model | CTR-AUC | CVR-AUC | CTCVR-AUC | PCOC |\n")
        f.write(f"|-------|---------|---------|-----------|------|\n")
        for model_type in model_types:
            r = all_results[model_type]
            f.write(f"| {model_type.upper()} | {r['ctr_auc']:.4f} | {r['cvr_auc']:.4f} | {r['ctcvr_auc']:.4f} | {r['pcoc']:.4f} |\n")
        
        f.write(f"\n## Relative Improvement (vs ESMM)\n\n")
        f.write(f"| Model | CVR-AUC Δ | CTCVR-AUC Δ |\n")
        f.write(f"|-------|-----------|-------------|\n")
        for model_type in model_types:
            if model_type == 'esmm':
                continue
            r = all_results[model_type]
            cvr_diff = r['cvr_auc'] - esmm_results['cvr_auc']
            ctcvr_diff = r['ctcvr_auc'] - esmm_results['ctcvr_auc']
            f.write(f"| {model_type.upper()} | {cvr_diff:+.4f} | {ctcvr_diff:+.4f} |\n")
    
    print(f"\nResults saved to ./results/baseline_comparison_{timestamp}.md")


if __name__ == '__main__':
    main()
