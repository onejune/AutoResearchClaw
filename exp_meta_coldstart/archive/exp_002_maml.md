# exp_002: Meta-Learning 实验结果

K-shot = 100

## 汇总

| 方法 | Mean AUC | Mean PCOC | 训练时间 | 说明 |
|------|----------|-----------|---------|------|
| MAML | 0.5148 | 46.213 | 24s | 二阶梯度 meta-learning |
| FOMAML | 0.5336 | 42.568 | 23s | 一阶近似 MAML，计算更高效 |
| ANIL | 0.5126 | 45.739 | 15s | 只在 head 层做 inner loop |
| ProtoNet | 0.5927 | 39.000 | 7s | 原型网络，embedding 距离分类 |

## 结论

- （实验完成后补充）
