# Multi-Grained ID 项目状态

**最后更新**: 2026-04-01 17:05  
**状态**: ⏸️ 已暂停（代码已更新，待运行）

---

## 当前进度

### ✅ 已完成

1. **项目框架搭建**
   - 目录结构创建完成
   - `README.md` 包含 6 个实验方向及论文索引
   - `requirements.txt` 依赖配置

2. **核心代码模块**
   - `src/data/loader.py`: IVR v16 数据加载器
     - ✅ 支持字符串特征自动编码
     - ✅ 支持层次化特征对（fine→coarse）
     - ✅ 支持随机划分 train/test
     - ⚠️ 待测试跨域划分（按 BT）
   
   - `src/train/trainer.py`: 统一训练器
     - ✅ 动态 GPU/CPU 切换
     - ✅ 混合精度训练 (AMP)
     - ✅ 梯度累积
   
   - `src/models/hierarchical.py`: 层次化 Embedding (exp003)
     - ✅ HierarchicalEmbedding 层
     - ✅ HierarchicalWideDeep 模型
     - ✅ 门控统计输出
   
   - `src/models/autoemb.py`: 自动化维度分配 (exp001)
     - ✅ AutoEmbEmbedding
     - ✅ AutoEmbWideDeep
     - ✅ 启发式维度推荐规则

3. **实验脚本**
   - `scripts/run_exp003_hierarchical.py`: exp003 实验脚本
     - ✅ Baseline vs Hierarchical 对比
     - ✅ 分 BT 评估
     - ✅ 门控权重分析
     - ⚠️ 未完整运行（暂停前正在训练）

4. **论文文档** (每个实验对应一个 paper.md)
   - ✅ `experiments/exp001_autoemb/paper.md` - AutoEmb (KDD 2021)
   - ✅ `experiments/exp002_dds/paper.md` - Data Distribution Search
   - ✅ `experiments/exp003_hierarchical/paper.md` - Hierarchical Embeddings
   - ✅ `experiments/exp004_metaemb/paper.md` - MetaEmb
   - ✅ `experiments/exp005_contrastive/paper.md` - Contrastive Learning
   - ✅ `experiments/exp006_fibinet/paper.md` - FiBiNET + AutoFIS

5. **实验设计文档**
   - ✅ `experiments/exp003_hierarchical/design.md`
   - ✅ `experiments/exp003_hierarchical/result.md` (模板)

---

## ⚠️ 待解决问题

### 1. 数据集使用规范（重要！）

**固定使用**:
- 训练集: `/mnt/data/oss_wanjun/pai_work/open_research/dataset/ivr_sample_v16_ctcvr_sample/train/`
- 测试集: `/mnt/data/oss_wanjun/pai_work/open_research/dataset/ivr_sample_v16_ctcvr_sample/test/`

**不要自己划分 train/test**，直接用现成的目录！

**特征处理**:
- 所有特征都是类别特征，已编码好
- 不做任何额外处理（不编码、不归一化）
- 跳过 `_merged_temp.parquet` 等临时文件

**标签**:
- `ctcvr_label`: ✅ 有（在 `part-*.parquet` 中）
- `click_label`: ✅ 有

---

### 2. 广告 ID 层级结构

由细到粗：
```
campaignid → campaignsetid → offerid → demand_pkgname → business_type
```

层次化特征对（用于 exp003）：
- `campaignid` → `campaignsetid`
- `offerid` → `demand_pkgname`
- `demand_pkgname` → `business_type`

---

### 3. exp003 实验未完成

**暂停时进度**: ~37% (Baseline 训练到一半)

**剩余工作**:
1. 完成 Baseline WideDeep 训练和评估
2. 训练 Hierarchical 模型
3. 对比分析并填写 `result.md`
4. 提交 git

---

## 📋 下一步计划 (恢复后)

### 优先级 1: 完成 exp003
- [ ] 重新运行 `scripts/run_exp003_hierarchical.py`
- [ ] 等待训练完成（预计 5-8 分钟）
- [ ] 分析结果并更新 `experiments/exp003_hierarchical/result.md`
- [ ] 提交 git: `exp: exp003 hierarchical embedding 初步结果`

### 优先级 2: 启动 exp001
- [ ] 创建 `scripts/run_exp001_autoemb.py`
- [ ] 对比 Fixed-64 vs AutoEmb (启发式)
- [ ] 分析参数量节省和 AUC 变化

### 优先级 3: 补充缺失模块
- [ ] `src/models/dds.py` (exp002)
- [ ] `src/models/metaemb.py` (exp004)
- [ ] `src/models/contrastive.py` (exp005)
- [ ] `src/models/fibinet.py` (exp006)
- [ ] `src/methods/contrastive_loss.py`
- [ ] `src/methods/meta_learner.py`

---

## 📝 重要记录

### 数据集统计 (IVR v16 CTCVR)
- Train: ~3.2M samples
- Test: ~1.2M samples
- CTR: 41.49%
- CTCVR: 11.37%
- Business Types: 19 个（字符串名称）

### 关键 Business Type
| BT 名称 | 样本量 | 特点 |
|--------|--------|------|
| shopee_cps | 1,097,581 | 最大，33.6% |
| shein | 488,377 | 第二大 |
| aedsp | 315,616 | - |
| aecps | 300,593 | - |
| lazada_cps | 278,231 | - |
| bk | 7,564 | 激励广告，极端 OOD |

### 实验配置
- Label: `ctcvr_label` (不是 `click_label`!)
- 划分：随机 80/20，seed=42
- Batch size: 512
- LR: 5e-5
- Epochs: 1 (所有 CTR/CVR 任务统一 1 epoch)

---

## 🔧 快速恢复命令

```bash
cd /mnt/workspace/open_research/autoresearch/multi_grained_id

# 重新运行 exp003
python scripts/run_exp003_hierarchical.py 2>&1 | tee /tmp/exp003_hierarchical.log

# 查看训练日志
tail -f /tmp/exp003_hierarchical.log

# 查看结果
cat experiments/exp003_hierarchical/result.md
cat results/exp003_hierarchical/results.json
```

---

**备注**: 下次启动时优先完成 exp003，不要同时开多个实验，避免资源竞争。
