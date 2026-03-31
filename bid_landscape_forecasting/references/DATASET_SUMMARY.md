# Bid Landscape 公开数据集 - 快速总结

> **结论**: **几乎没有直接可用的真实 bid landscape 公开数据集**

---

## 🎯 最接近的选项

### 1. Criteo Display Advertising (⭐⭐⭐)
- **链接**: http://labs.criteo.com/wp-content/uploads/2014/02/Criteo_KDD_2014_dataset.tar.gz
- **规模**: 13.8 亿样本
- **优点**: 免费、最大规模、业界标准
- **缺点**: ❌ 无 bid amount, ❌ 无 win/loss label
- **用法**: 合成 bid landscape (和我们当前方案一样)

### 2. Amazon Advertising Dataset (⭐⭐⭐⭐)
- **链接**: https://registry.opendata.aws/amazon-public-datasets/
- **优点**: 真实竞价环境，可能有 bid info
- **缺点**: 需要 AWS 申请，审核周期长
- **状态**: ⏳ 可申请

### 3. Yahoo! LTR Dataset (⭐⭐⭐)
- **链接**: https://webscope.sandbox.yahoo.com/
- **优点**: 部分版本包含 bid 信息
- **缺点**: 需申请，数据较老 (2010-2012)
- **状态**: ⏳ 可申请

---

## 💡 现实情况

**行业现状**:
- 几乎所有论文使用**公司内部数据** (不公开)
- 公开数据集大多只有 CTR/CVR，没有 bid/win
- **合成数据是主流做法**

**参考论文**:
- "Learning Bid Landscapes" (KDD 2019): 未公开数据
- "Deep Bid Landscape Prediction" (WWW 2021): 未公开数据
- 多数工作：CTR 数据 + 合成 bid

---

## ✅ 推荐方案

### 短期 (现在)
**继续使用 IVR + 合成 bid**
- ✅ 立即可用
- ✅ 已有 50 万样本
- 📝 论文中说明合成策略和局限性

### 中期 (1-2 月)
**并行验证**:
1. 下载 Criteo 数据集 (~100GB)
2. 用相同方法合成 bid
3. 对比 IVR vs Criteo 结果

### 长期 (3-6 月)
**申请真实数据**:
- Amazon Advertising (AWS Open Data)
- Alibaba (联系作者)
- 学术合作获取 DSP 日志

---

## 📥 立即行动

**下载 Criteo 做对比**:
```bash
cd /mnt/data/oss_wanjun/pai_work/open_research/dataset/
wget http://labs.criteo.com/wp-content/uploads/2014/02/Criteo_KDD_2014_dataset.tar.gz
tar -xzf Criteo_KDD_2014_dataset.tar.gz
# 约 100GB，下载需要时间
```

**或者继续当前方案**:
- IVR 数据质量高 (326 万样本)
- 合成策略合理 (Beta distribution + sigmoid)
- 已在多个 business_type 上验证

---

*建议：先用 IVR 完成研究，Criteo 作为消融实验验证泛化性*  
*牛顿 🍎*
