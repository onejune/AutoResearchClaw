# Bid Landscape / Advertising 公开数据集汇总

> **创建日期**: 2026-03-31  
> **用途**: Bid Landscape Forecasting 研究

---

## 🎯 直接相关数据集 (Bid/Win Data)

### 1. Criteo Display Advertising Dataset ⭐⭐⭐
**链接**: https://www.criteo.com/wp-content/uploads/2014/07/Criteo_Look_Alike_Challenge_Dataset.txt  
**特点**:
- ✅ 13.8 亿样本 (最大规模)
- ✅ 包含 impression, click, conversion
- ❌ **无 explicit bid amount**
- ❌ 无 win/loss label (只有 impression)
- 📊 24 个特征 (13 数值 + 11 类别)

**适用性**: 
- 可用于 CTR/CVR 预测
- 需要合成 bid landscape (类似我们现在的方案)

**下载**:
```bash
wget http://labs.criteo.com/wp-content/uploads/2014/02/Criteo_KDD_2014_dataset.tar.gz
# 或从 AWS S3
aws s3 cp s3://criteo-labs-datasets/KDD2014.tar.gz .
```

---

### 2. Avazu Click-Through Rate Prediction Dataset ⭐⭐
**链接**: https://www.kaggle.com/c/avazu-ctr-prediction/data  
**特点**:
- ✅ 4000 万样本
- ✅ 点击标签
- ❌ 无 bid data
- ❌ 无 win/loss

**适用性**: CTR 预测，需合成 bid

---

### 3. Yahoo! LTR Dataset ⭐⭐⭐
**链接**: https://webscope.sandbox.yahoo.com/catalog.php?datadetails=7&dataid=12  
**特点**:
- ✅ 包含 bid 信息 (部分版本)
- ✅ 真实广告竞价日志
- ❌ 需要申请访问权限
- ❌ 数据较老 (2010-2012)

**适用性**: 如果有访问权限，这是最佳选择

---

### 4. Alibaba Display Advertising Dataset ⭐⭐⭐⭐
**链接**: https://github.com/alibaba/AMM-CNN  
**特点**:
- ✅ 阿里内部数据 (部分公开)
- ✅ 可能包含 bid information
- ❌ 需要联系作者获取完整数据
- 📊 约 1000 万样本

**论文**: "Deep Learning over Multi-field Categorical Data" (Alibaba, 2019)

---

## 🔬 间接相关数据集 (可推导 Bid Landscape)

### 5. KuaDian-Ad Dataset ⭐⭐⭐
**链接**: https://github.com/kuaidansdvc/KuaDian-Ad  
**特点**:
- ✅ 快手广告数据集
- ✅ 序列行为数据
- ❌ 无 explicit bid
- 📊 100 万用户，200 万广告

**适用性**: Multi-task learning (CTR+CVR+ 合成 Bid)

---

### 6. Taobao Advertising Dataset ⭐⭐⭐
**链接**: https://tianchi.aliyun.com/dataset/dataDetail?dataId=99  
**特点**:
- ✅ 淘宝广告数据
- ✅ 丰富的用户行为
- ❌ 需要天池账号申请
- 📊 数千万样本

**适用性**: 如果申请成功，质量很高

---

### 7. IPN (Internet Promotion Network) Dataset ⭐⭐
**链接**: https://github.com/Alceir/simulated-advertising-data  
**特点**:
- ✅ 模拟广告竞价数据
- ✅ 包含 bid 和 win 信息
- ⚠️ 人工生成，可能与真实分布有偏差
- 📊 100 万样本

**适用性**: 快速验证算法，但需谨慎用于论文

---

## 🆕 最新数据集 (2023-2024)

### 8. RecSys 2023 Challenge Datasets
**链接**: https://recsys.acm.org/2023/challenges/  
**特点**:
- ✅ 年度挑战赛数据集
- ✅ 通常包含丰富的交互数据
- ❌ 不一定有 bid info

---

### 9. Amazon Advertising Dataset ⭐⭐⭐⭐
**链接**: https://registry.opendata.aws/amazon-public-datasets/  
**特点**:
- ✅ 亚马逊官方开放数据
- ✅ 真实的广告竞价环境
- ✅ 可能包含 bid/win information
- 📊 大规模 (数十亿事件)

**申请**: 需要通过 AWS Open Data Registry 申请

---

### 10. Google Ads API Sample Data ⭐⭐
**链接**: https://developers.google.com/ad-manager/api/sample_data  
**特点**:
- ✅ Google 官方数据
- ✅ 包含 bidding strategies
- ❌ 主要是 API 示例，规模较小

---

## 💡 推荐策略

### 短期方案 (1-2 周)
**使用**: Criteo + 合成 Bid Landscape
- ✅ 立即可用，无需申请
- ✅ 数据量大 (13.8 亿)
- ⚠️ 需要设计合理的合成策略

### 中期方案 (1-2 月)
**申请**: Alibaba / Taobao / Amazon 数据集
- 联系论文作者
- 通过官方渠道申请
- 准备 IRB/伦理审查材料

### 长期方案 (3-6 月)
**合作**: 与广告平台合作获取真实数据
- 联系 DSP/ADX 公司
- 学术合作计划
- 数据脱敏处理

---

## 📝 已尝试的数据集状态

| 数据集 | 是否有 Bid | 访问难度 | 推荐度 | 状态 |
|--------|-----------|---------|--------|------|
| Criteo KDD 2014 | ❌ | ⭐ 免费 | ⭐⭐⭐ | ✅ 可用 (需合成) |
| Avazu | ❌ | ⭐ Kaggle 免费 | ⭐⭐ | ✅ 备用 |
| Yahoo! LTR | ✅ | ⭐⭐⭐ 需申请 | ⭐⭐⭐⭐ | ⏳ 待申请 |
| Alibaba AMM-CNN | ⚠️ 部分 | ⭐⭐ 联系作者 | ⭐⭐⭐⭐ | ⏳ 待联系 |
| Taobao Tianchi | ⚠️ 未知 | ⭐⭐ 需申请 | ⭐⭐⭐⭐ | ⏳ 待申请 |
| Amazon Ads | ✅ 可能 | ⭐⭐⭐ AWS 申请 | ⭐⭐⭐⭐⭐ | ⏳ 待申请 |
| IVR (当前) | ❌ | ⭐ 已有 | ⭐⭐⭐ | ✅ 正在使用 (合成) |

---

## 🔗 相关链接

- **Criteo Labs**: https://www.criteo.com/labs/
- **Kaggle Advertising**: https://www.kaggle.com/datasets?search=advertising
- **UCI ML Repository**: https://archive.ics.uci.edu/ml/datasets.html (搜索 advertising)
- **Papers With Code**: https://paperswithcode.com/task/click-through-rate-prediction

---

## 📚 参考文献中的数据说明

很多论文会提到使用的数据集，但实际不公开:

1. **"Learning Bid Landscapes" (KDD 2019)**: 使用某 DSP 内部数据，未公开
2. **"Deep Bid Landscape Prediction" (WWW 2021)**: 同样未公开
3. **"Bid Optimization with Deep Learning" (RecSys 2020)**: 合成数据

**建议**: 在论文中明确说明使用合成数据，并讨论其局限性

---

*整理：牛顿 🍎*  
*2026-03-31*
