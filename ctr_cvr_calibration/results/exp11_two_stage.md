# 实验 11: Two-Stage Calibration

> **实验日期**: 2026-03-31 17:08:47

## 方法对比

| Method | AUC | ECE | PCOC | Temperature |
|--------|-----|-----|------|-------------|
| Baseline | 0.8010 | 0.007615 | 1.0172 | - |
| Temperature Only | 0.8010 | 0.007613 | 1.0172 | 1.0014 |
| Isotonic Only | 0.8012 | 0.000000 | 1.0000 | - |
| Two-Stage | 0.8012 | 0.000004 | 1.0000 | 1.0014 |

## 核心发现

- Two-Stage vs Iso Only: 基本不变
- Temperature 参数：1.0014
- 两步校准的优势：无明显优势

---

*实验报告 - 牛顿 🍎*
