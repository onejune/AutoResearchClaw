"""
ChorusCVR 完整实验脚本
在全量数据上训练和评估
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

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.chorus_cvr import ChorusCVR, ESMM
from src.losses.chorus_loss import ChorusCVRLoss, ESMMLoss


def load_data(max_samples=None):
    """加载数据"""
    train_path = '/mnt/data/oss_wanjun/pai_work/open_research/dataset/ali_ccp/ali_ccp_train.csv'
    test_path = '/mnt/data/oss_wanjun/pai_work/open_research/dataset/ali_ccp/ali_ccp_test.csv'
    
    print(f"Loading data...")
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
                batch_size=4096, epochs=1, lr=1e-3, device='cpu'):
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
            
            if (i + 1) % 50 == 0:
                pbar.set_postfix({'loss': f"{total_loss / (i+1):.4f}"})
        
        print(f"Epoch {epoch+1} - Avg Loss: {total_loss / n_batches:.4f}")
    
    return model


@torch.no_grad()
def evaluate_model(model, test_df, sparse_features, dense_features, device='cpu', batch_size=8192):
    """评估模型 - 分批处理避免内存问题"""
    model.eval()
    
    all_pCTR = []
    all_pCVR = []
    all_pCTCVR = []
    all_click_labels = []
    all_conversion_labels = []
    
    n_samples = len(test_df)
    n_batches = (n_samples + batch_size - 1) // batch_size
    
    for i in tqdm(range(n_batches), desc='Evaluating'):
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
    
    # 拼接
    all_pCTR = np.concatenate(all_pCTR)
    all_pCVR = np.concatenate(all_pCVR)
    all_pCTCVR = np.concatenate(all_pCTCVR)
    all_click_labels = np.concatenate(all_click_labels)
    all_conversion_labels = np.concatenate(all_conversion_labels)
    
    # 计算指标
    results = {}
    
    # CTR AUC
    results['ctr_auc'] = roc_auc_score(all_click_labels, all_pCTR)
    
    # CVR AUC (点击空间)
    click_mask = all_click_labels > 0.5
    if click_mask.sum() > 0 and all_conversion_labels[click_mask].sum() > 0:
        results['cvr_auc'] = roc_auc_score(all_conversion_labels[click_mask], all_pCVR[click_mask])
    else:
        results['cvr_auc'] = 0.5
    
    # CTCVR AUC (全空间)
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


def run_experiment(model_type, train_df, test_df, sparse_features, dense_features, 
                   sparse_feature_dims, device='cpu', batch_size=4096, lr=1e-3):
    """运行单个实验"""
    print(f"\n{'='*60}")
    print(f"Training {model_type.upper()}")
    print(f"{'='*60}")
    
    if model_type == 'esmm':
        model = ESMM(
            sparse_feature_dims=sparse_feature_dims,
            dense_feature_num=len(dense_features),
            embedding_dim=16,
            shared_hidden_dims=[256, 128],
            tower_hidden_dims=[64, 32],
            dropout_rate=0.1
        )
        loss_fn = ESMMLoss()
    else:  # chorus
        model = ChorusCVR(
            sparse_feature_dims=sparse_feature_dims,
            dense_feature_num=len(dense_features),
            embedding_dim=16,
            shared_hidden_dims=[256, 128],
            tower_hidden_dims=[64, 32],
            dropout_rate=0.1
        )
        loss_fn = ChorusCVRLoss(
            loss_weights={
                'ctcvr': 1.0,
                'cvr_ipw': 1.0,
                'ctuncvr': 1.0,
                'uncvr_ipw': 1.0,
                'align_ipw': 0.5,  # 降低对齐损失权重
            }
        )
    
    # 训练
    model = train_model(model, loss_fn, train_df, sparse_features, dense_features,
                        batch_size=batch_size, epochs=1, lr=lr, device=device)
    
    # 评估
    results = evaluate_model(model, test_df, sparse_features, dense_features, device)
    
    print(f"\n{model_type.upper()} Results:")
    for k, v in results.items():
        print(f"  {k}: {v:.4f}")
    
    return results, model


def main():
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_samples', type=int, default=500000, help='Max training samples')
    parser.add_argument('--batch_size', type=int, default=4096, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--device', type=str, default='cpu', help='Device')
    args = parser.parse_args()
    
    print("="*60)
    print("ChorusCVR Full Experiment")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)
    
    # 加载数据
    train_df, test_df, sparse_features, dense_features, sparse_feature_dims = load_data(args.max_samples)
    
    # 划分验证集
    val_size = int(len(train_df) * 0.1)
    val_df = train_df.iloc[-val_size:]
    train_df = train_df.iloc[:-val_size]
    
    print(f"\nTrain: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    
    # 运行 ESMM 基线
    esmm_results, esmm_model = run_experiment(
        'esmm', train_df, test_df, sparse_features, dense_features,
        sparse_feature_dims, device=args.device, batch_size=args.batch_size, lr=args.lr
    )
    
    # 运行 ChorusCVR
    chorus_results, chorus_model = run_experiment(
        'chorus', train_df, test_df, sparse_features, dense_features,
        sparse_feature_dims, device=args.device, batch_size=args.batch_size, lr=args.lr
    )
    
    # 对比结果
    print("\n" + "="*60)
    print("Final Comparison")
    print("="*60)
    print(f"{'Metric':<15} {'ESMM':<12} {'ChorusCVR':<12} {'Diff':<12}")
    print("-"*51)
    for k in ['ctr_auc', 'cvr_auc', 'ctcvr_auc', 'pcoc']:
        diff = chorus_results[k] - esmm_results[k]
        sign = '+' if diff > 0 else ''
        print(f"{k:<15} {esmm_results[k]:<12.4f} {chorus_results[k]:<12.4f} {sign}{diff:.4f}")
    
    # 保存结果
    os.makedirs('./results', exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    with open(f'./results/exp_full_{timestamp}.md', 'w') as f:
        f.write(f"# ChorusCVR Full Experiment Results\n\n")
        f.write(f"**Time**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"## Configuration\n")
        f.write(f"- Train samples: {len(train_df)}\n")
        f.write(f"- Test samples: {len(test_df)}\n")
        f.write(f"- Batch size: {args.batch_size}\n")
        f.write(f"- Learning rate: {args.lr}\n")
        f.write(f"- Epochs: 1\n\n")
        f.write(f"## Results\n\n")
        f.write(f"| Metric | ESMM | ChorusCVR | Diff |\n")
        f.write(f"|--------|------|-----------|------|\n")
        for k in ['ctr_auc', 'cvr_auc', 'ctcvr_auc', 'pcoc']:
            diff = chorus_results[k] - esmm_results[k]
            sign = '+' if diff > 0 else ''
            f.write(f"| {k} | {esmm_results[k]:.4f} | {chorus_results[k]:.4f} | {sign}{diff:.4f} |\n")
        
        f.write(f"\n## Analysis\n\n")
        if chorus_results['cvr_auc'] > esmm_results['cvr_auc']:
            f.write(f"ChorusCVR 在 CVR-AUC 上提升了 {(chorus_results['cvr_auc'] - esmm_results['cvr_auc'])*100:.2f} 个百分点。\n")
        if chorus_results['ctcvr_auc'] > esmm_results['ctcvr_auc']:
            f.write(f"ChorusCVR 在 CTCVR-AUC 上提升了 {(chorus_results['ctcvr_auc'] - esmm_results['ctcvr_auc'])*100:.2f} 个百分点。\n")
    
    print(f"\nResults saved to ./results/exp_full_{timestamp}.md")
    
    # 保存模型
    torch.save(esmm_model.state_dict(), './results/esmm_model.pt')
    torch.save(chorus_model.state_dict(), './results/chorus_model.pt')
    print("Models saved.")


if __name__ == '__main__':
    main()
