"""
ChorusCVR 超参数搜索实验
优先级：
1. align_ipw 权重调整
2. IPW 裁剪范围
3. 模型容量
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
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.chorus_cvr import ChorusCVR
from src.losses.chorus_loss import ChorusCVRLoss


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
        train_df_shuffled = train_df.sample(frac=1).reset_index(drop=True)
        
        pbar = tqdm(range(n_batches), desc=f'{desc} Epoch {epoch+1}', ncols=100)
        for i in pbar:
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, n_samples)
            batch_df = train_df_shuffled.iloc[start_idx:end_idx]
            
            sparse_tensors, dense_tensor, click_labels, conversion_labels = create_batch(
                batch_df, sparse_features, dense_features, device
            )
            
            optimizer.zero_grad()
            outputs = model(sparse_tensors, dense_tensor)
            loss, _ = loss_fn(outputs, click_labels, conversion_labels)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
    
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


def run_experiment(config, train_df, test_df, sparse_features, dense_features, sparse_feature_dims, device='cpu'):
    """运行单个实验"""
    model = ChorusCVR(
        sparse_feature_dims=sparse_feature_dims,
        dense_feature_num=len(dense_features),
        embedding_dim=config['embedding_dim'],
        shared_hidden_dims=config['shared_hidden_dims'],
        tower_hidden_dims=config['tower_hidden_dims'],
        dropout_rate=config['dropout_rate'],
    )
    
    loss_fn = ChorusCVRLoss(
        loss_weights=config['loss_weights'],
        ipw_clip_min=config['ipw_clip_min'],
        ipw_clip_max=config['ipw_clip_max'],
    )
    
    model = train_model(
        model, loss_fn, train_df, sparse_features, dense_features,
        batch_size=config['batch_size'], epochs=1, lr=config['lr'],
        device=device, desc=config['name']
    )
    
    results = evaluate_model(model, test_df, sparse_features, dense_features, device)
    return results


def main():
    print("="*70)
    print("ChorusCVR Hyperparameter Search")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)
    
    # 加载数据 (用 100w 样本加速搜索)
    train_df, test_df, sparse_features, dense_features, sparse_feature_dims = load_data(1000000)
    
    device = 'cpu'
    
    # 基础配置
    base_config = {
        'embedding_dim': 16,
        'shared_hidden_dims': [256, 128],
        'tower_hidden_dims': [64, 32],
        'dropout_rate': 0.1,
        'batch_size': 4096,
        'lr': 1e-3,
        'loss_weights': {
            'ctcvr': 1.0,
            'cvr_ipw': 1.0,
            'ctuncvr': 1.0,
            'uncvr_ipw': 1.0,
            'align_ipw': 0.5,
        },
        'ipw_clip_min': 0.01,
        'ipw_clip_max': 1.0,
    }
    
    all_results = []
    
    # ============================================================
    # 实验1: align_ipw 权重搜索
    # ============================================================
    print("\n" + "="*70)
    print("Exp 1: align_ipw Weight Search")
    print("="*70)
    
    align_weights = [0.0, 0.1, 0.5, 1.0, 2.0]
    
    for align_w in align_weights:
        config = base_config.copy()
        config['name'] = f'align={align_w}'
        config['loss_weights'] = {
            'ctcvr': 1.0, 'cvr_ipw': 1.0, 'ctuncvr': 1.0, 
            'uncvr_ipw': 1.0, 'align_ipw': align_w
        }
        
        results = run_experiment(config, train_df, test_df, sparse_features, 
                                 dense_features, sparse_feature_dims, device)
        results['config'] = config['name']
        results['exp_group'] = 'align_weight'
        all_results.append(results)
        
        print(f"  {config['name']}: CVR={results['cvr_auc']:.4f}, CTCVR={results['ctcvr_auc']:.4f}, PCOC={results['pcoc']:.4f}")
    
    # ============================================================
    # 实验2: IPW 裁剪范围搜索
    # ============================================================
    print("\n" + "="*70)
    print("Exp 2: IPW Clip Range Search")
    print("="*70)
    
    ipw_clips = [(0.01, 1.0), (0.05, 1.0), (0.1, 1.0), (0.1, 0.9), (0.2, 0.8)]
    
    for clip_min, clip_max in ipw_clips:
        config = base_config.copy()
        config['name'] = f'ipw=[{clip_min},{clip_max}]'
        config['ipw_clip_min'] = clip_min
        config['ipw_clip_max'] = clip_max
        
        results = run_experiment(config, train_df, test_df, sparse_features, 
                                 dense_features, sparse_feature_dims, device)
        results['config'] = config['name']
        results['exp_group'] = 'ipw_clip'
        all_results.append(results)
        
        print(f"  {config['name']}: CVR={results['cvr_auc']:.4f}, CTCVR={results['ctcvr_auc']:.4f}, PCOC={results['pcoc']:.4f}")
    
    # ============================================================
    # 实验3: 模型容量搜索
    # ============================================================
    print("\n" + "="*70)
    print("Exp 3: Model Capacity Search")
    print("="*70)
    
    model_configs = [
        {'emb': 16, 'shared': [256, 128], 'tower': [64, 32], 'name': 'small'},
        {'emb': 32, 'shared': [256, 128], 'tower': [64, 32], 'name': 'emb32'},
        {'emb': 16, 'shared': [512, 256, 128], 'tower': [64, 32], 'name': 'deep_shared'},
        {'emb': 16, 'shared': [256, 128], 'tower': [128, 64, 32], 'name': 'deep_tower'},
        {'emb': 32, 'shared': [512, 256, 128], 'tower': [128, 64, 32], 'name': 'large'},
    ]
    
    for mc in model_configs:
        config = base_config.copy()
        config['name'] = mc['name']
        config['embedding_dim'] = mc['emb']
        config['shared_hidden_dims'] = mc['shared']
        config['tower_hidden_dims'] = mc['tower']
        
        results = run_experiment(config, train_df, test_df, sparse_features, 
                                 dense_features, sparse_feature_dims, device)
        results['config'] = config['name']
        results['exp_group'] = 'model_capacity'
        all_results.append(results)
        
        print(f"  {config['name']}: CVR={results['cvr_auc']:.4f}, CTCVR={results['ctcvr_auc']:.4f}, PCOC={results['pcoc']:.4f}")
    
    # ============================================================
    # 汇总结果
    # ============================================================
    print("\n" + "="*70)
    print("Summary - Best Configs per Group")
    print("="*70)
    
    df_results = pd.DataFrame(all_results)
    
    for group in ['align_weight', 'ipw_clip', 'model_capacity']:
        group_df = df_results[df_results['exp_group'] == group]
        best_idx = group_df['cvr_auc'].idxmax()
        best = group_df.loc[best_idx]
        print(f"\n{group}:")
        print(f"  Best: {best['config']}")
        print(f"  CVR-AUC: {best['cvr_auc']:.4f}, CTCVR-AUC: {best['ctcvr_auc']:.4f}, PCOC: {best['pcoc']:.4f}")
    
    # 保存结果
    os.makedirs('./results', exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # 保存 JSON
    with open(f'./results/hyperparam_search_{timestamp}.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # 保存 Markdown
    with open(f'./results/hyperparam_search_{timestamp}.md', 'w') as f:
        f.write(f"# ChorusCVR Hyperparameter Search Results\n\n")
        f.write(f"**Time**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"**Data**: Train={len(train_df)}, Test={len(test_df)}\n\n")
        
        for group in ['align_weight', 'ipw_clip', 'model_capacity']:
            f.write(f"\n## {group.replace('_', ' ').title()}\n\n")
            f.write("| Config | CVR-AUC | CTCVR-AUC | PCOC |\n")
            f.write("|--------|---------|-----------|------|\n")
            
            group_df = df_results[df_results['exp_group'] == group]
            for _, row in group_df.iterrows():
                f.write(f"| {row['config']} | {row['cvr_auc']:.4f} | {row['ctcvr_auc']:.4f} | {row['pcoc']:.4f} |\n")
    
    print(f"\nResults saved to ./results/hyperparam_search_{timestamp}.md")
    
    # 返回最佳组合
    print("\n" + "="*70)
    print("Recommended Best Config")
    print("="*70)
    
    # 找到 CVR-AUC 最高的配置
    best_overall = df_results.loc[df_results['cvr_auc'].idxmax()]
    print(f"Best overall: {best_overall['config']}")
    print(f"CVR-AUC: {best_overall['cvr_auc']:.4f}")
    print(f"CTCVR-AUC: {best_overall['ctcvr_auc']:.4f}")
    print(f"PCOC: {best_overall['pcoc']:.4f}")


if __name__ == '__main__':
    main()
