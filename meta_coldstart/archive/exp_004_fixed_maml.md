# exp_004: 修复后 Meta-Learning 实验结果

K-shot = 100

## 汇总

| 方法 | Mean AUC | Mean PCOC | 训练时间 | 说明 |
|------|----------|-----------|---------|------|
| MAML | 0.5210 | 28.207 | 4s | 一阶近似 MAML（FOMAML），计算更高效 |
| ANIL | 0.4609 | 43.233 | 3s | 只在 head 层做 inner loop |
| ProtoNet | 0.5674 | 43.112 | 1s | 原型网络，embedding 距离分类 |

## 结论

- （实验完成后补充）
