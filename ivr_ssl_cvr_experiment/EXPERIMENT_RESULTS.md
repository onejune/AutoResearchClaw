# IVR SSL 对比学习实验结果

## 实验概述
- **项目**：IVR 自监督对比学习 (P10)
- **目标**：在 DSP 场景下使用对比学习增强 CVR 预估模型的特征表示能力
- **数据集**：ivr_sample_v16_ctcvr（326万训练，123万测试，126个类别特征）

## 实验方法对比
1. **Baseline** (DeepFM): AUC = 0.8484
2. **SimGCL**: AUC = 0.8476 (-0.80‰)
3. **SupCon**: AUC = 0.8490 (**+0.06‰**, 最优)
4. **DomainCL**: AUC = 0.8471 (-1.30‰)
5. **FeatureMask**: AUC = 0.8483 (-0.10‰)
6. **DirectAU**: AUC = 0.8467 (-1.70‰)
7. **参数调优 SupCon**: AUC = 0.8488 (Temp=0.2, CL_Weight=0.1)
8. **混合方法**: AUC = 0.8457 (-0.27‰)
9. **MoCo**: AUC = 0.8483 (-0.10‰)

## 关键技术突破
- 解决了 GPU 训练数值稳定性问题（DataLoader + Xavier 初始化 + 梯度裁剪 + Dropout）
- 实现了多种对比学习方法的统一框架
- 完成了 SupCon 参数调优（温度系数和CL权重网格搜索）

## 重要发现
- SupCon 是最优方法，但相比 Baseline 提升仅 0.06‰
- 对比学习在当前数据集上收益有限，可能是由于 DeepFM 已经是相对有效的模型架构
- 参数调优对性能有一定影响，最佳参数组合为 Temp=0.2, CL_Weight=0.1
- 复杂方法（MoCo、混合）表现不如简单方法

## 代码结构
- `scripts/train_ivr_ssl_all_methods.py`: 所有对比学习方法的统一实现
- `scripts/train_supcon_param_tuning.py`: SupCon 参数调优脚本
- `scripts/train_hybrid_ssl.py`: 混合对比学习方法
- `scripts/train_moco_ssl.py`: MoCo 对比学习实现
- `scripts/analyze_ivr_ssl_detailed.py`: 详细分析脚本
- `scripts/analyze_results_summary.py`: 结果汇总分析