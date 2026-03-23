# CVR Delayed Feedback Modeling — KDD 2026

## 课题
电商广告 CVR 预估中的延迟反馈建模：灵活分布方法与在线标签修正

## 方法
FDAM（Flexible Distribution Approach with online label correction）：
- 用 Weibull 分布建模转化延迟，比指数分布更灵活
- 在线更新分布参数
- 对未转化样本做软标签修正

## 数据集
Criteo Conversion Logs（KDD 2014，Chapelle et al.）
- 路径：`/mnt/data/oss_wanjun/pai_work/open_research/dataset/criteo_dataset/data.txt`
- 约 1589 万行，tab 分隔

## 实验结果

### 500k 样本（快速验证）
| 模型 | AUC |
|------|-----|
| Naive | 0.7030 |
| DFM | 0.7106 |
| ES-DFM | 0.6770 |
| **FDAM（ours）** | **0.6948** |

FDAM vs ES-DFM: **+1.79 千分点**（p=0.032）

### 全量数据（待更新）
见 `results/metrics_fulldata.md`

## 运行方式
```bash
cd experiment/
python main.py
```

## 论文
见 `paper/paper_revised.md` 和 `paper_revised.tex`
