# 实验记录索引

每个实验一个子目录，包含：
- `design.md`：实验设计（目标、配置、假设）
- `result.md`：实验结果（AUC、PCOC、分BT表格、结论）
- `metrics.json`：机器可读的数字结果

原始输出文件（模型权重、CSV）存放在 `../results/` 下。

---

## 实验列表

| ID | 名称 | 状态 | 核心结论 |
|----|------|------|---------|
| [exp001](exp001_baseline/) | WideDeep Baseline | ✅ 完成 | OOD AUC 最低 0.50，PCOC 跨 BT 差异巨大，问题确认 |
| [exp002](exp002_feature_importance/) | 特征重要性分析 | ✅ 完成 | bucket 特征跨域稳定，购买行为特征虚假相关 |
| exp003 | IRM 多环境训练 | 🔜 待做 | - |
| exp004 | 因果特征子集 vs 全特征 | 🔜 待做 | - |
| exp005 | NAS 特征交叉搜索 | 🔜 待做 | - |
