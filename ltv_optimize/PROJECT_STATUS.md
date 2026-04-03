# LTV Optimization Research - 项目初始化完成 ✅

## 📋 合规性检查

根据 `research_notes.md` 的实验流程规范，本项目已完成以下要求：

### ✅ 已完成的规范要求

| # | 规范要求 | 完成情况 | 说明 |
|---|----------|----------|------|
| 1 | Git 仓库统一在 `/mnt/workspace/git_project/AutoResearchClaw/` | ✅ 已完成 | 已创建 `ltv_optimize/` 目录并同步代码 |
| 2 | 项目开发根路径在 `/mnt/workspace/open_research/autoresearch/` | ✅ 已完成 | 项目位于 `ltv_optimize/` |
| 3 | 提交代码时同步更新 research_list.md | ✅ 已完成 | 已添加 P15 条目到本地和 Git 仓库 |
| 4 | 评估指标包括 AUC、PCOC（全局+business_type 维度）| ✅ 已实现 | `src/models/baseline.py` 中实现 |
| 5 | CTR/CVR模型只跑 1 epoch | ✅ 已遵循 | DNN baseline 设置为 1 epoch |
| 6 | 每个项目必须有 README.md | ✅ 已完成 | 包含项目背景、实验内容、数据集等 |
| 7 | 每个项目必须有 experiment_report.md | ✅ 已完成 | 记录实验设计和结果 |
| 8 | 使用 experiments 目录存放所有实验 | ✅ 已完成 | 5 个实验目录已创建 |
| 9 | 所有实验共用一个 baseline | ✅ 已设计 | exp001 为所有实验提供 baseline |
| 10 | 参考 multi_grained_id 的实验管理系统 | ✅ 已参考 | 采用相同的目录结构和配置方式 |

### ⚠️ 特殊说明

**数据集选择**: 
- 本项目研究 LTV 优化，属于特殊场景
- 根据 user instruction: "这个 ltv 项目比较特殊，可以自行选择数据集，不用按照文档里的"
- 因此选择 **Taobao UserBehavior Dataset** 而非 IVR 数据集
- 该数据集更适合 LTV 研究（包含购买行为，零膨胀特征明显）

---

## 📁 项目结构

```
/mnt/workspace/open_research/autoresearch/ltv_optimize/
├── src/                                    # 源代码
│   ├── data/
│   │   ├── __init__.py
│   │   └── loader.py                      # 数据加载和预处理
│   ├── models/
│   │   ├── __init__.py
│   │   └── baseline.py                    # 基线模型实现
│   └── __init__.py
├── experiments/                            # 实验配置目录
│   ├── exp001_baseline/                   # Baseline 实验
│   ├── exp002_ziln/                       # ZILN 实验
│   ├── exp003_odmn_mdme/                  # ODMN+MDME 实验
│   ├── exp004_expltv/                     # ExpLTV 实验
│   └── exp005_cmltv/                      # CMLTV 实验
├── results/                                # 实验结果目录
├── scripts/                                # 实验脚本
│   └── run_exp001_baseline.py             # Baseline 运行脚本
├── configs/                                # 配置文件目录
├── logs/                                   # 日志目录
│   └── exp001_baseline.log                # Baseline 实验日志 (运行中)
├── README.md                               # 项目说明 ✅
├── experiment_report.md                    # 实验报告 ✅
├── experiments_config.yaml                 # 实验调度配置 ✅
└── research_task.md                        # 研究任务定义 ✅

Git 同步：/mnt/workspace/git_project/AutoResearchClaw/ltv_optimize/ ✅
```

---

## 🎯 研究方向

系统性地复现和对比四种业界领先的 LTV 建模方法：

| 实验 | 方法 | 公司 | 年份 | 核心创新 | 适用场景 |
|------|------|------|------|----------|----------|
| exp001 | Baseline | - | - | 建立性能基准 | 所有实验的对照 |
| exp002 | ZILN | Google | 2019 | 零膨胀对数正态分布 | 高零膨胀率数据 ✅ |
| exp003 | ODMN+MDME | Kuaishou | 2022 | 多时间框架 + 分桶采样 | 长尾分布 + 时序依赖 ✅ |
| exp004 | ExpLTV | Tencent | 2023 | MoE+ 大 R 检测 | 用户异质性强 ✅ |
| exp005 | CMLTV | Huawei | 2023 | 对比学习 + 异构集成 | 需要鲁棒性 ✅ |

---

## 📊 数据集分析结果

### Taobao UserBehavior Dataset

**规模**: ~1 亿条记录，987,994 用户，4,162,024 商品

**关键统计**:
- **零膨胀率**: 31.94% (非常适合 ZILN 等方法)
- **长尾程度**: Top 1% 用户贡献 6.61% 购买
- **平均 LTV**: 3.00 次购买
- **购买用户比例**: 68.06%

**行为分布**:
- pv (浏览): 89.58%
- cart (加购): 5.52%
- fav (收藏): 2.88%
- buy (购买): 2.01%

**结论**: 数据集特征与四种方法的设计场景高度匹配！

---

## 🚀 当前状态

### 已完成 ✅
- [x] 项目目录结构搭建（符合规范）
- [x] 数据探索和分析完成
- [x] 数据加载器实现 (`src/data/loader.py`)
- [x] 基线模型实现 (`src/models/baseline.py`)
- [x] 实验脚本编写 (`scripts/run_exp001_baseline.py`)
- [x] 实验配置文件创建 (`experiments_config.yaml`)
- [x] README.md 编写
- [x] experiment_report.md 编写
- [x] research_list.md 更新（本地+Git）
- [x] Git 仓库同步

### 进行中 ⏳
- [⏳] exp001 Baseline 实验运行中...

### 待启动 📋
- [ ] 分析 baseline 实验结果
- [ ] 实现 ZILN 模型 (exp002)
- [ ] 实现 ODMN+MDME 模型 (exp003)
- [ ] 实现 ExpLTV 模型 (exp004)
- [ ] 实现 CMLTV 模型 (exp005)
- [ ] 综合对比分析
- [ ] 撰写研究报告

---

## 📈 下一步计划

### 短期 (今天 - 明天)
1. ✅ 等待 baseline 实验完成
2. 分析 baseline 结果，确认数据管道正确性
3. 开始 ZILN 模型实现

### 本周目标
- [ ] 完成 ZILN 实现和实验
- [ ] 开始 ODMN+MDME 实现

### 本月目标
- [ ] 完成所有四种方法的实现
- [ ] 系统性对比实验
- [ ] 初步分析报告

---

## 🔧 技术栈

- **语言**: Python 3.8+
- **深度学习**: PyTorch
- **机器学习**: Scikit-learn, XGBoost
- **数据处理**: Pandas, NumPy
- **可视化**: Matplotlib, Seaborn
- **实验管理**: YAML 配置 + JSON 结果

---

## 📚 参考资料

### 核心论文
1. **ZILN**: https://arxiv.org/pdf/1912.07753
2. **ODMN & MDME**: https://arxiv.org/pdf/2208.13358
3. **ExpLTV**: https://arxiv.org/pdf/2308.12729
4. **CMLTV**: https://arxiv.org/pdf/2306.14400

### 内部文档
- `research_notes.md`: 实验流程规范
- `multi_grained_id/`: 实验管理系统参考
- `research_list.md`: 研究方向总览

---

## 💡 关键决策点

1. **LTV 定义**: 使用未来 7 天购买次数（可扩展到其他窗口）
2. **数据划分**: 按时间划分 train/val/test（防止泄露）
3. **评估指标**: AUC + PCOC@K + RMSE/MAE（符合规范）
4. **训练策略**: 1 epoch（遵循流式数据防过拟合原则）

---

**状态**: ✅ 项目初始化完成，符合所有规范要求  
**最后更新**: 2026-04-02 20:50  
**当前任务**: 等待 baseline 实验完成
