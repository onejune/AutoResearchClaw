"""
ChorusCVR 快速测试脚本
"""
import os
import sys
import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.chorus_cvr import ChorusCVR, ESMM
from src.losses.chorus_loss import ChorusCVRLoss, ESMMLoss


def load_data_simple(max_samples=50000):
    """简化的数据加载"""
    train_path = '/mnt/data/oss_wanjun/pai_work/open_research/dataset/ali_ccp/ali_ccp_train.csv'
    test_path = '/mnt/data/oss_wanjun/pai_work/open_research/dataset/ali_ccp/ali_ccp_test.csv'
    
    print(f"Loading data (max_samples={max_samples})...")
    train_df = pd.read_csv(train_path, nrows=max_samples)
    test_df = pd.read_csv(test_path, nrows=max_samples // 5 if max_samples else None)
    
    # 特征定义
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
    
    # 填充缺失值
    train_df[dense_features] = train_df[dense_features].fillna(0)
    test_df[dense_features] = test_df[dense_features].fillna(0)
    
    print(f"Train: {len(train_df)}, Test: {len(test_df)}")
    print(f"Click rate: {train_df['click'].mean():.4f}")
    print(f"Purchase rate: {train_df['purchase'].mean():.4f}")
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
                batch_size=2048, epochs=1, lr=1e-3, device='cpu'):
    """训练模型"""
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    
    n_samples = len(train_df)
    n_batches = (n_samples + batch_size - 1) // batch_size
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        # Shuffle
        train_df_shuffled = train_df.sample(frac=1).reset_index(drop=True)
        
        pbar = tqdm(range(n_batches), desc=f'Epoch {epoch+1}')
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
            pbar.set_postfix({'loss': f"{total_loss / (i+1):.4f}"})
        
        print(f"Epoch {epoch+1} - Avg Loss: {total_loss / n_batches:.4f}")
    
    return model


@torch.no_grad()
def evaluate_model(model, test_df, sparse_features, dense_features, device='cpu'):
    """评估模型"""
    model.eval()
    
    sparse_tensors, dense_tensor, click_labels, conversion_labels = create_batch(
        test_df, sparse_features, dense_features, device
    )
    
    outputs = model(sparse_tensors, dense_tensor)
    
    # 转换为 numpy
    pCTR = outputs['pCTR'].cpu().numpy()
    pCVR = outputs['pCVR'].cpu().numpy()
    pCTCVR = outputs['pCTCVR'].cpu().numpy()
    click_labels = click_labels.cpu().numpy()
    conversion_labels = conversion_labels.cpu().numpy()
    
    # 计算指标
    results = {}
    
    # CTR AUC
    results['ctr_auc'] = roc_auc_score(click_labels, pCTR)
    
    # CVR AUC (点击空间)
    click_mask = click_labels > 0.5
    if click_mask.sum() > 0 and conversion_labels[click_mask].sum() > 0:
        results['cvr_auc'] = roc_auc_score(conversion_labels[click_mask], pCVR[click_mask])
    else:
        results['cvr_auc'] = 0.5
    
    # CTCVR AUC (全空间)
    ctcvr_labels = click_labels * conversion_labels
    if ctcvr_labels.sum() > 0:
        results['ctcvr_auc'] = roc_auc_score(ctcvr_labels, pCTCVR)
    else:
        results['ctcvr_auc'] = 0.5
    
    # PCOC
    if click_mask.sum() > 0:
        actual_cvr = conversion_labels[click_mask].mean()
        pred_cvr = pCVR[click_mask].mean()
        results['pcoc'] = pred_cvr / actual_cvr if actual_cvr > 0 else 1.0
    else:
        results['pcoc'] = 1.0
    
    return results


def main():
    device = 'cpu'  # 使用 CPU 避免 CUDA 问题
    max_samples = 100000  # 10万样本快速测试
    
    print("="*60)
    print("ChorusCVR Quick Test")
    print("="*60)
    
    # 加载数据
    train_df, test_df, sparse_features, dense_features, sparse_feature_dims = load_data_simple(max_samples)
    
    # 划分验证集
    val_size = int(len(train_df) * 0.1)
    val_df = train_df.iloc[-val_size:]
    train_df = train_df.iloc[:-val_size]
    
    print(f"\nTrain: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    
    # 测试 ESMM 基线
    print("\n" + "="*60)
    print("Training ESMM (Baseline)")
    print("="*60)
    
    esmm_model = ESMM(
        sparse_feature_dims=sparse_feature_dims,
        dense_feature_num=len(dense_features),
        embedding_dim=16,
        shared_hidden_dims=[256, 128],
        tower_hidden_dims=[64, 32],
        dropout_rate=0.1
    )
    
    esmm_loss_fn = ESMMLoss()
    esmm_model = train_model(esmm_model, esmm_loss_fn, train_df, sparse_features, dense_features,
                             batch_size=2048, epochs=1, lr=1e-3, device=device)
    
    esmm_results = evaluate_model(esmm_model, test_df, sparse_features, dense_features, device)
    print("\nESMM Results:")
    for k, v in esmm_results.items():
        print(f"  {k}: {v:.4f}")
    
    # 测试 ChorusCVR
    print("\n" + "="*60)
    print("Training ChorusCVR")
    print("="*60)
    
    chorus_model = ChorusCVR(
        sparse_feature_dims=sparse_feature_dims,
        dense_feature_num=len(dense_features),
        embedding_dim=16,
        shared_hidden_dims=[256, 128],
        tower_hidden_dims=[64, 32],
        dropout_rate=0.1
    )
    
    chorus_loss_fn = ChorusCVRLoss()
    chorus_model = train_model(chorus_model, chorus_loss_fn, train_df, sparse_features, dense_features,
                               batch_size=2048, epochs=1, lr=1e-3, device=device)
    
    chorus_results = evaluate_model(chorus_model, test_df, sparse_features, dense_features, device)
    print("\nChorusCVR Results:")
    for k, v in chorus_results.items():
        print(f"  {k}: {v:.4f}")
    
    # 对比结果
    print("\n" + "="*60)
    print("Comparison")
    print("="*60)
    print(f"{'Metric':<15} {'ESMM':<10} {'ChorusCVR':<10} {'Diff':<10}")
    print("-"*45)
    for k in ['ctr_auc', 'cvr_auc', 'ctcvr_auc', 'pcoc']:
        diff = chorus_results[k] - esmm_results[k]
        print(f"{k:<15} {esmm_results[k]:<10.4f} {chorus_results[k]:<10.4f} {diff:+.4f}")
    
    # 保存结果
    os.makedirs('./results', exist_ok=True)
    with open('./results/quick_test_results.md', 'w') as f:
        f.write("# ChorusCVR Quick Test Results\n\n")
        f.write(f"## Data\n")
        f.write(f"- Train samples: {len(train_df)}\n")
        f.write(f"- Test samples: {len(test_df)}\n\n")
        f.write(f"## Results\n\n")
        f.write(f"| Metric | ESMM | ChorusCVR | Diff |\n")
        f.write(f"|--------|------|-----------|------|\n")
        for k in ['ctr_auc', 'cvr_auc', 'ctcvr_auc', 'pcoc']:
            diff = chorus_results[k] - esmm_results[k]
            f.write(f"| {k} | {esmm_results[k]:.4f} | {chorus_results[k]:.4f} | {diff:+.4f} |\n")
    
    print("\nResults saved to ./results/quick_test_results.md")


if __name__ == '__main__':
    main()
