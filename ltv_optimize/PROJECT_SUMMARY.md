# LTV Optimization Research - Project Summary

## 项目启动状态 ✅

**启动时间**: 2026-04-02 20:10  
**当前阶段**: 数据探索与准备  
**负责人**: Algorithm Engineer Agent

## 已完成工作

### 1. 项目初始化 ✅
- [x] 创建项目目录结构
- [x] 编写 README.md (项目概述)
- [x] 编写 experiment_report.md (实验报告模板)
- [x] 编写 implementation_plan.md (详细实施计划)
- [x] 配置 AutoResearchClaw (config.arc.yaml)
- [x] 创建研究任务文档 (research_task.md, research_prompt.md)

### 2. 数据集选择 ✅
**选定数据集**: Taobao UserBehavior Dataset

**关键统计信息** (初步):
- **总交互数**: ~1 亿条记录
- **用户数**: 987,994
- **商品数**: 4,162,024
- **行为类型**: pv (89.58%), cart (5.52%), fav (2.88%), buy (2.01%)
- **购买用户**: 672,404 (68.06%)
- **零膨胀率**: ~32% (非购买用户比例)

**选择理由**:
1. 包含完整的行为漏斗 (浏览→收藏/加购→购买)
2. 显著的零膨胀特征，适合测试 ZILN 等方法
3. 大量用户和交互，可支持深度学习模型
4. 有时间戳，可构建多时间窗口特征

### 3. 研究工具准备 ⏳
- [x] AutoResearchClaw 已安装 (`/mnt/workspace/open_research/AutoResearchClaw_tool`)
- [x] 配置文件已创建
- [⏳] 数据探索脚本运行中
- [ ] 待：LLM API 配置完善后启动全自动文献调研

## 研究方向

本研究将系统性地复现和对比四种业界领先的 LTV 建模方法:

| 方法 | 公司 | 年份 | 核心思想 | 适用场景 |
|------|------|------|----------|----------|
| **ZILN** | Google | 2019 | 零膨胀对数正态分布 | 高度零膨胀数据 |
| **ODMN+MDME** | Kuaishou | 2022 | 多时间框架 + 分桶采样 | 长尾分布 + 时序依赖 |
| **ExpLTV** | Tencent | 2023 | MoE + 大 R 检测 | 用户异质性强的场景 |
| **CMLTV** | Huawei | 2023 | 对比学习 + 异构集成 | 需要鲁棒性的场景 |

## 项目目录结构

```
/mnt/workspace/open_research/autoresearch/ltv_optimize/
├── README.md                      # 项目说明 ✅
├── experiment_report.md           # 实验报告 ✅
├── implementation_plan.md         # 实施计划 ✅
├── research_task.md              # 研究任务 ✅
├── research_prompt.md            # 研究提示 ✅
├── config.arc.yaml               # AutoResearchClaw 配置 ✅
├── 研究方向.txt                  # 原始研究方向 ✅
├── data/                         # 数据处理模块 (待开发)
├── models/                       # 模型实现
│   ├── baseline/                # 基线模型
│   ├── ziln/                    # ZILN 实现
│   ├── odmn_mdme/               # ODMN+MDME 实现
│   ├── expltv/                  # ExpLTV 实现
│   └── cmltv/                   # CMLTV 实现
├── training/                     # 训练相关代码
├── evaluation/                   # 评估指标
├── experiments/                  # 实验配置和结果
├── scripts/                      # 辅助脚本
│   └── explore_data.py          # 数据探索脚本 ✅
└── logs/                         # 日志和中间结果
    ├── data_exploration.log     # 数据探索日志 (运行中)
    └── researchclaw_log.txt     # AutoResearchClaw 日志
```

## 下一步计划

### 短期 (本周)
1. **完成数据探索** - 分析 LTV 分布、长尾程度、时间模式
2. **搭建基线框架** - 实现数据加载、特征工程、评估管道
3. **实现简单基线** - Linear Regression, XGBoost, Two-stage model
4. **开始 ZILN 实现** - 最基础的零膨胀模型

### 中期 (2-4 周)
1. **完成所有方法实现** - ZILN → ODMN/MDME → ExpLTV → CMLTV
2. **系统性实验** - 统一数据集、统一指标、多次随机种子
3. **消融研究** - 理解各组件贡献

### 长期 (6-8 周)
1. **深度分析** - 可视化、案例研究、错误分析
2. **报告撰写** - 形成完整的研究报告
3. **代码整理** - 开源级别的文档和测试

## 关键技术挑战

1. **零膨胀处理**: 32% 的非购买用户需要特殊建模
2. **长尾分布**: 少数大 R 用户贡献大部分价值
3. **特征工程**: 如何从行为序列中提取有效特征
4. **评估指标**: AUC、PCOC、Log-Likelihood 的平衡
5. **计算资源**: 深度学习模型需要 GPU 加速

## 预期成果

1. **代码库**: 四种方法的 PyTorch 实现
2. **基准测试**: 首个 unified benchmark for LTV methods
3. **实证洞察**: 什么数据特征适合什么方法
4. **研究报告**: 系统性对比分析和改进建议

## 参考资料

### 核心论文
1. **ZILN**: https://arxiv.org/pdf/1912.07753
2. **ODMN & MDME**: https://arxiv.org/pdf/2208.13358
3. **ExpLTV**: https://arxiv.org/pdf/2308.12729
4. **CMLTV**: https://arxiv.org/pdf/2306.14400

### 相关工具
- **AutoResearchClaw**: `/mnt/workspace/open_research/AutoResearchClaw_tool`
- **PyTorch**: Deep learning framework
- **XGBoost/LightGBM**: Tree-based baselines
- **Pandas/NumPy**: Data processing

## 团队沟通

- **进度更新**: 每周提供实验进展报告
- **问题反馈**: 遇到技术障碍及时提出
- **决策点**: 重要设计选择需要确认

---

*最后更新：2026-04-02 20:25*  
*状态：数据探索进行中，预计 1-2 小时内完成初步分析*
