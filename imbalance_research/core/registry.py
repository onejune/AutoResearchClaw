"""
实验注册表 - 所有实验配置集中管理

添加新实验只需在这里注册，不需要创建额外文件
"""

from typing import Dict, Any

# 需要排除的特征 (防止数据泄露或过拟合)
EXCLUDE_FEATURES = [
    'deviceid',  # 用户ID，会导致模型记住用户
]

# 基础配置模板
_BASE_CONFIG = {
    'data': {
        'dir': '/mnt/data/oss_wanjun/pai_work/open_research/dataset/ivr_sample_v16_ctcvr_sample/',
        'label_col': 'ctcvr_label',
        'sample_ratio': 0.05,  # 5% 数据用于快速验证
        'exclude_features': EXCLUDE_FEATURES,
    },
    'model': {
        'type': 'mlp',
        'hidden_dims': [256, 128, 64],
        'dropout': 0.3,
        'embed_dim': 16,
    },
    'training': {
        'epochs': 20,
        'batch_size': 256,
        'learning_rate': 1e-3,
        'patience': 5,
        'num_workers': 4,
    },
    'gpu': {
        'device': -1,  # -1 = 自动选择
    }
}


def _merge_config(base: Dict, override: Dict) -> Dict:
    """深度合并配置"""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _merge_config(result[key], value)
        else:
            result[key] = value
    return result


def _create_config(override: Dict) -> Dict:
    """基于模板创建配置"""
    return _merge_config(_BASE_CONFIG, override)


# ============================================================
# 实验注册表
# ============================================================

EXPERIMENTS: Dict[str, Dict[str, Any]] = {}


def register_experiment(name: str, config: Dict[str, Any], description: str = ""):
    """注册实验"""
    full_config = _create_config(config)
    full_config['description'] = description
    EXPERIMENTS[name] = full_config


# ============================================================
# Baseline 实验
# ============================================================

register_experiment(
    name='baseline_bce',
    description='MLP + BCE Loss (基准)',
    config={
        'loss': {'type': 'bce'},
    }
)

register_experiment(
    name='baseline_focal',
    description='MLP + Focal Loss (标准配置)',
    config={
        'loss': {
            'type': 'focal',
            'gamma': 2.0,
            'alpha': 0.25,
        },
    }
)

register_experiment(
    name='baseline_widedeep_bce',
    description='Wide&Deep + BCE Loss',
    config={
        'model': {'type': 'widedeep'},
        'loss': {'type': 'bce'},
    }
)

register_experiment(
    name='baseline_widedeep_focal',
    description='Wide&Deep + Focal Loss',
    config={
        'model': {'type': 'widedeep'},
        'loss': {
            'type': 'focal',
            'gamma': 2.0,
            'alpha': 0.25,
        },
    }
)

# ============================================================
# Gamma 参数扫描
# ============================================================

for gamma in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]:
    register_experiment(
        name=f'focal_gamma_{gamma}'.replace('.', '_'),
        description=f'Focal Loss gamma={gamma}',
        config={
            'loss': {
                'type': 'focal',
                'gamma': gamma,
                'alpha': 0.25,
            },
        }
    )

# ============================================================
# Alpha 参数扫描
# ============================================================

for alpha in [0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5]:
    register_experiment(
        name=f'focal_alpha_{alpha}'.replace('.', '_'),
        description=f'Focal Loss alpha={alpha}',
        config={
            'loss': {
                'type': 'focal',
                'gamma': 2.0,
                'alpha': alpha,
            },
        }
    )

# ============================================================
# Focal Loss 变体
# ============================================================

register_experiment(
    name='balanced_focal',
    description='Balanced Focal Loss (自动调整 alpha)',
    config={
        'loss': {
            'type': 'balanced',
            'gamma': 2.0,
        },
    }
)

register_experiment(
    name='asymmetric_focal',
    description='Asymmetric Focal Loss (正负样本不同 gamma)',
    config={
        'loss': {
            'type': 'asymmetric',
            'gamma_pos': 2.0,
            'gamma_neg': 1.0,
            'alpha': 0.25,
        },
    }
)

register_experiment(
    name='dynamic_focal',
    description='Dynamic Focal Loss (gamma 随训练衰减)',
    config={
        'loss': {
            'type': 'dynamic',
            'gamma_init': 3.0,
            'gamma_end': 1.0,
        },
    }
)

register_experiment(
    name='smoothed_focal',
    description='Focal Loss + Label Smoothing',
    config={
        'loss': {
            'type': 'smoothed',
            'gamma': 2.0,
            'alpha': 0.25,
            'epsilon': 0.1,
        },
    }
)


# ============================================================
# 实用函数
# ============================================================

def get_experiment(name: str) -> Dict[str, Any]:
    """获取实验配置"""
    if name not in EXPERIMENTS:
        raise ValueError(f"未知实验: {name}. 可用: {list(EXPERIMENTS.keys())}")
    return EXPERIMENTS[name]


def list_experiments() -> list:
    """列出所有实验名"""
    return list(EXPERIMENTS.keys())
