"""
使用超参搜索得到的最佳配置在 500w 样本上运行实验
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

from src.models.chorus_cvr import ChorusCVR
from src.models.baselines import ESMM, ESCM2
from src.losses.chorus_loss import ChorusCVRLoss
from src.losses.baseline_losses import ESMMLoss, ESCM2Loss


def load_data(max_samples=None):
    train_path = '/mnt/data/oss_wanjun/pai_work/open_research/dataset/ali_ccp/ali_ccp_train.csv'
    test_path = '/mnt/data/oss_wanjun/pai_work/open_research/dataset/ali_ccp/ali_ccp_test.csv'
    
    print(f"Loading data (max_samples={max_samples})...")
    train_df = pd.read_csv(train_path, nrows=max_samples)
    test_df = pd.read_csv(test_path, nrows=max_samples // 5 if max_samples else None)
    
    sparse_features = ['101', '121', '122', '124', '125', '126', '127', '128', '129',
                       '205', '206', '207', '210', '216', '508', '509', '702', '853', '301']
    dense_features = ['D109_14', 'D110_14', 'D127_14', 'D150_14', 'D508', 'D509', 'D702', 'D853']
    
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
    sparse_tensors = {
        feat: torch.tensor(df[feat].values, dtype=torch.long, device=device)
        for feat in sparse_features
    }
    dense_tensor = torch.tensor(df[dense_features].values, dtype=torch.float32, device=device)
    click_labels = torch.tensor(df['click'].values, dtype=torch.float32, device=device)
    conversion_labels = torch.tensor(df['purchase'].values, dtype=torch.float32, device=device)
    
    return sparse_tensors, dense_tensor, click_labels, conversion_labels


def train_model(model, loss_fn, train_df, sparse_features, dense_features, 
                batch_size=4096, epochs=1, lr=1e-3, device='cpu', desc='Training'):
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    
    n_samples = len(train_df)
    n_batches = (n_samples + batch_size - 1) // batch_size
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        train_df_shuffled = train_df.sample(frac=1).reset_index(drop=True)
        
        pbar = tqdm(range(n_batches), desc=f'{desc} Epoch {epoch+1}')
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
        
        print(f"{desc} Epoch {epoch+1} - Avg Loss: {total_loss / n_batches:.4f}")
    
    return model


@torch.no_grad()
def evaluate_model(model, test_df, sparse_features, dense_features, device='cpu', batch_size=8192):
    model.eval()
    
    all_pCTR, all_pCVR, all_pCTCVR = [], [], []
    all_click_labels, all_conversion_labels = [], []
    
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
    results['ctr_auc'] = roc_auc_score(all_click_labels, all_pCTR)
    
    click_mask = all_click_labels > 0.5
    if click_mask.sum() > 0 and all_conversion_labels[click_mask].sum() > 0:
        results['cvr_auc'] = roc_auc_score(all_conversion_labels[click_mask], all_pCVR[click_mask])
    else:
        results['cvr_auc'] = 0.5
    
    ctcvr_labels = all_click_labels * all_conversion_labels
    if ctcvr_labels.sum() > 0:
        results['ctcvr_auc'] = roc_auc_score(ctcvr_labels, all_pCTCVR)
    else:
        results['ctcvr_auc'] = 0.5
    
    if click_mask.sum() > 0:
        actual_cvr = all_conversion_labels[click_mask].mean()
        pred_cvr = all_pCVR[click_mask].mean()
        results['pcoc'] = pred_cvr / actual_cvr if actual_cvr > 0 else 1.0
    else:
        results['pcoc'] = 1.0
    
    return results


def main():
    print("="*70)
    print("ChorusCVR Best Config Experiment (5M samples)")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)
    
    # 加载数据
    train_df, test_df, sparse_features, dense_features, sparse_feature_dims = load_data(5000000)
    
    device = 'cpu'
    batch_size = 4096
    lr = 1e-3
    
    all_results = {}
    
    # ============================================================
    # 1. ESMM Baseline
    # ============================================================
    print("\n" + "="*70)
    print("Training ESMM (Baseline)")
    print("="*70)
    
    model_esmm = ESMM(
        sparse_feature_dims=sparse_feature_dims,
        dense_feature_num=len(dense_features),
        embedding_dim=16,
        shared_hidden_dims=[256, 128],
        tower_hidden_dims=[64, 32],
        dropout_rate=0.1,
    )
    loss_fn_esmm = ESMMLoss()
    
    model_esmm = train_model(model_esmm, loss_fn_esmm, train_df, sparse_features, 
                             dense_features, batch_size, 1, lr, device, 'ESMM')
    all_results['ESMM'] = evaluate_model(model_esmm, test_df, sparse_features, dense_features, device)
    
    # ============================================================
    # 2. ESCM2-IPW (Best Baseline)
    # ============================================================
    print("\n" + "="*70)
    print("Training ESCM2-IPW")
    print("="*70)
    
    model_escm2 = ESCM2(
        sparse_feature_dims=sparse_feature_dims,
        dense_feature_num=len(dense_features),
        embedding_dim=16,
        shared_hidden_dims=[256, 128],
        tower_hidden_dims=[64, 32],
        dropout_rate=0.1,
        mode='ipw',
    )
    loss_fn_escm2 = ESCM2Loss(mode='ipw')
    
    model_escm2 = train_model(model_escm2, loss_fn_escm2, train_df, sparse_features, 
                              dense_features, batch_size, 1, lr, device, 'ESCM2-IPW')
    all_results['ESCM2-IPW'] = evaluate_model(model_escm2, test_df, sparse_features, dense_features, device)
    
    # ============================================================
    # 3. ChorusCVR (Original Config)
    # ============================================================
    print("\n" + "="*70)
    print("Training ChorusCVR (Original Config)")
    print("="*70)
    
    model_chorus_orig = ChorusCVR(
        sparse_feature_dims=sparse_feature_dims,
        dense_feature_num=len(dense_features),
        embedding_dim=16,
        shared_hidden_dims=[256, 128],
        tower_hidden_dims=[64, 32],
        dropout_rate=0.1,
    )
    loss_fn_chorus_orig = ChorusCVRLoss(
        loss_weights={'ctcvr': 1.0, 'cvr_ipw': 1.0, 'ctuncvr': 1.0, 'uncvr_ipw': 1.0, 'align_ipw': 0.5},
        ipw_clip_min=0.01, ipw_clip_max=1.0,
    )
    
    model_chorus_orig = train_model(model_chorus_orig, loss_fn_chorus_orig, train_df, sparse_features, 
                                    dense_features, batch_size, 1, lr, device, 'ChorusCVR-Orig')
    all_results['ChorusCVR-Orig'] = evaluate_model(model_chorus_orig, test_df, sparse_features, dense_features, device)
    
    # ============================================================
    # 4. ChorusCVR (Best Config: align=0, deep_tower)
    # ============================================================
    print("\n" + "="*70)
    print("Training ChorusCVR (Best Config)")
    print("="*70)
    
    model_chorus_best = ChorusCVR(
        sparse_feature_dims=sparse_feature_dims,
        dense_feature_num=len(dense_features),
        embedding_dim=16,
        shared_hidden_dims=[256, 128],
        tower_hidden_dims=[128, 64, 32],  # deeper tower
        dropout_rate=0.1,
    )
    loss_fn_chorus_best = ChorusCVRLoss(
        loss_weights={'ctcvr': 1.0, 'cvr_ipw': 1.0, 'ctuncvr': 1.0, 'uncvr_ipw': 1.0, 'align_ipw': 0.0},  # no align
        ipw_clip_min=0.01, ipw_clip_max=1.0,
    )
    
    model_chorus_best = train_model(model_chorus_best, loss_fn_chorus_best, train_df, sparse_features, 
                                    dense_features, batch_size, 1, lr, device, 'ChorusCVR-Best')
    all_results['ChorusCVR-Best'] = evaluate_model(model_chorus_best, test_df, sparse_features, dense_features, device)
    
    # ============================================================
    # 打印结果
    # ============================================================
    print("\n" + "="*70)
    print("Final Results (5M samples)")
    print("="*70)
    
    print(f"\n{'Model':<20} {'CTR-AUC':<12} {'CVR-AUC':<12} {'CTCVR-AUC':<12} {'PCOC':<12}")
    print("-"*68)
    for name, r in all_results.items():
        print(f"{name:<20} {r['ctr_auc']:<12.4f} {r['cvr_auc']:<12.4f} {r['ctcvr_auc']:<12.4f} {r['pcoc']:<12.4f}")
    
    # 相对提升
    esmm = all_results['ESMM']
    print(f"\n{'Model':<20} {'CVR Δ':<12} {'CTCVR Δ':<12}")
    print("-"*44)
    for name, r in all_results.items():
        if name == 'ESMM':
            continue
        cvr_diff = r['cvr_auc'] - esmm['cvr_auc']
        ctcvr_diff = r['ctcvr_auc'] - esmm['ctcvr_auc']
        print(f"{name:<20} {cvr_diff:+.4f}      {ctcvr_diff:+.4f}")
    
    # 保存结果
    os.makedirs('./results', exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    with open(f'./results/best_config_{timestamp}.md', 'w') as f:
        f.write(f"# ChorusCVR Best Config Results\n\n")
        f.write(f"**Time**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"**Data**: Train={len(train_df)}, Test={len(test_df)}\n\n")
        
        f.write("## Results\n\n")
        f.write("| Model | CTR-AUC | CVR-AUC | CTCVR-AUC | PCOC |\n")
        f.write("|-------|---------|---------|-----------|------|\n")
        for name, r in all_results.items():
            f.write(f"| {name} | {r['ctr_auc']:.4f} | {r['cvr_auc']:.4f} | {r['ctcvr_auc']:.4f} | {r['pcoc']:.4f} |\n")
        
        f.write("\n## Relative Improvement (vs ESMM)\n\n")
        f.write("| Model | CVR-AUC Δ | CTCVR-AUC Δ |\n")
        f.write("|-------|-----------|-------------|\n")
        for name, r in all_results.items():
            if name == 'ESMM':
                continue
            cvr_diff = r['cvr_auc'] - esmm['cvr_auc']
            ctcvr_diff = r['ctcvr_auc'] - esmm['ctcvr_auc']
            f.write(f"| {name} | {cvr_diff:+.4f} | {ctcvr_diff:+.4f} |\n")
        
        f.write("\n## Best Config Details\n\n")
        f.write("```python\n")
        f.write("# ChorusCVR-Best config\n")
        f.write("loss_weights = {\n")
        f.write("    'ctcvr': 1.0, 'cvr_ipw': 1.0,\n")
        f.write("    'ctuncvr': 1.0, 'uncvr_ipw': 1.0,\n")
        f.write("    'align_ipw': 0.0,  # Key: disable alignment loss\n")
        f.write("}\n")
        f.write("tower_hidden_dims = [128, 64, 32]  # Deeper tower\n")
        f.write("```\n")
    
    print(f"\nResults saved to ./results/best_config_{timestamp}.md")


if __name__ == '__main__':
    main()
