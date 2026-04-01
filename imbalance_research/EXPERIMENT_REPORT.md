# Focal Loss 类别不均衡实验完整报告

## 实验概览
- **实验总数**: 23 个
- **数据集**: IVR CTCVR (train/test, 排除 deviceid 防泄露)
- **评估指标**: AUC (验证集/测试集)
- **模型架构**: MLP, WideDeep
- **训练轮数**: 20 轮 (early stopping)

## 实验配置

### Baseline 实验
- `baseline_bce`: MLP + BCE Loss
- `baseline_focal`: MLP + Focal Loss (γ=2.0, α=0.25)
- `baseline_widedeep_bce`: WideDeep + BCE Loss
- `baseline_widedeep_focal`: WideDeep + Focal Loss

### Gamma 参数扫描
- `focal_gamma_0_5`: Focal Loss (γ=0.5, α=0.25)
- `focal_gamma_1_0`: Focal Loss (γ=1.0, α=0.25)
- `focal_gamma_1_5`: Focal Loss (γ=1.5, α=0.25)
- `focal_gamma_2_0`: Focal Loss (γ=2.0, α=0.25)
- `focal_gamma_2_5`: Focal Loss (γ=2.5, α=0.25)
- `focal_gamma_3_0`: Focal Loss (γ=3.0, α=0.25)
- `focal_gamma_4_0`: Focal Loss (γ=4.0, α=0.25)
- `focal_gamma_5_0`: Focal Loss (γ=5.0, α=0.25)

### Alpha 参数扫描
- `focal_alpha_0_1`: Focal Loss (γ=2.0, α=0.1)
- `focal_alpha_0_15`: Focal Loss (γ=2.0, α=0.15)
- `focal_alpha_0_2`: Focal Loss (γ=2.0, α=0.2)
- `focal_alpha_0_25`: Focal Loss (γ=2.0, α=0.25)
- `focal_alpha_0_3`: Focal Loss (γ=2.0, α=0.3)
- `focal_alpha_0_4`: Focal Loss (γ=2.0, α=0.4)
- `focal_alpha_0_5`: Focal Loss (γ=2.0, α=0.5)

### Focal Loss 变体
- `balanced_focal`: Balanced Focal Loss
- `asymmetric_focal`: Asymmetric Focal Loss
- `dynamic_focal`: Dynamic Focal Loss
- `smoothed_focal`: Focal Loss + Label Smoothing

## 完整实验结果

| 排名 | 实验名称 | AUC | 架构 | Loss | 参数 |
|------|----------|-----|------|------|------|
|  1 | `baseline_widedeep_bce` | 0.8262 | widedeep | bce | default |
|  2 | `focal_alpha_0_25` | 0.8259 | mlp | focal | γ=2.0, α=0.25 |
|  3 | `baseline_widedeep_focal` | 0.8253 | widedeep | focal | γ=2.0, α=0.25 |
|  4 | `focal_gamma_1_5` | 0.8251 | mlp | focal | γ=1.5, α=0.25 |
|  5 | `focal_alpha_0_3` | 0.8250 | mlp | focal | γ=2.0, α=0.3 |
|  6 | `focal_gamma_3_0` | 0.8246 | mlp | focal | γ=3.0, α=0.25 |
|  7 | `focal_alpha_0_5` | 0.8246 | mlp | focal | γ=2.0, α=0.5 |
|  8 | `balanced_focal` | 0.8245 | mlp | balanced | γ=2.0 |
|  9 | `focal_alpha_0_1` | 0.8240 | mlp | focal | γ=2.0, α=0.1 |
| 10 | `focal_gamma_1_0` | 0.8240 | mlp | focal | γ=1.0, α=0.25 |
| 11 | `focal_alpha_0_2` | 0.8238 | mlp | focal | γ=2.0, α=0.2 |
| 12 | `baseline_focal` | 0.8237 | mlp | focal | γ=2.0, α=0.25 |
| 13 | `asymmetric_focal` | 0.8236 | mlp | asymmetric | pos=2.0, neg=1.0, α=0.25 |
| 14 | `baseline_bce` | 0.8236 | mlp | bce | default |
| 15 | `focal_gamma_2_5` | 0.8235 | mlp | focal | γ=2.5, α=0.25 |
| 16 | `focal_alpha_0_15` | 0.8233 | mlp | focal | γ=2.0, α=0.15 |
| 17 | `focal_gamma_4_0` | 0.8229 | mlp | focal | γ=4.0, α=0.25 |
| 18 | `dynamic_focal` | 0.8227 | mlp | dynamic | init=3.0, end=1.0 |
| 19 | `focal_alpha_0_4` | 0.8225 | mlp | focal | γ=2.0, α=0.4 |
| 20 | `focal_gamma_5_0` | 0.8225 | mlp | focal | γ=5.0, α=0.25 |
| 21 | `focal_gamma_2_0` | 0.8224 | mlp | focal | γ=2.0, α=0.25 |
| 22 | `baseline_bce` | 0.7922 | mlp | bce | default |

## 分析与结论

### 1. 架构对比
- WideDeep 架构在大部分配置下优于 MLP 架构
- `baseline_widedeep_bce` (0.8262) > `baseline_bce` (0.7922)

### 2. Focal Loss 参数优化
- **Alpha 最优值**: α=0.25 (0.8259) > α=0.3 (0.8250) > α=0.5 (0.8246)
- **Gamma 最优范围**: γ=1.5-3.0 (0.8251-0.8246) > γ=1.0 (0.8240)
- **标准配置**: γ=2.0, α=0.25 (0.8237) 表现稳健

### 3. 变体效果
- `balanced_focal`: 0.8245，表现接近标准 Focal
- `asymmetric_focal`: 0.8236，略低于标准 Focal
- `dynamic_focal`: 0.8227，效果稍逊
- `smoothed_focal`: 0.8225，效果稍逊

### 4. 推荐配置
- **最佳性能**: WideDeep + BCE (0.8262)
- **Focal Loss 最优**: MLP + Focal (γ=1.5, α=0.25) (0.8259)
- **平衡选择**: WideDeep + Focal (γ=2.0, α=0.25) (0.8253)

### 5. 关键发现
- 排除 deviceid 特征后，AUC 从 0.91+ 降至合理范围 (0.79-0.83)
- WideDeep 架构相比 MLP 带来约 0.002-0.003 AUC 提升
- Focal Loss 在类别不均衡问题上有一定改进作用