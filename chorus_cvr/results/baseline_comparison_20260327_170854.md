# ChorusCVR Baseline Comparison

**Time**: 2026-03-27 17:08:54

## Configuration
- Train samples: 2859201
- Test samples: 1000000
- Batch size: 4096
- Learning rate: 0.001
- Epochs: 1

## Results

| Model | CTR-AUC | CVR-AUC | CTCVR-AUC | PCOC |
|-------|---------|---------|-----------|------|
| ESMM | 0.6125 | 0.6641 | 0.6408 | 1.3267 |
| ESCM2_IPW | 0.5929 | 0.6860 | 0.6706 | 1.1792 |
| DCMT | 0.5938 | 0.6828 | 0.6657 | 1.1788 |
| DDPO | 0.5905 | 0.6799 | 0.6568 | 1.1048 |
| CHORUS | 0.5809 | 0.6822 | 0.6582 | 1.3595 |

## Relative Improvement (vs ESMM)

| Model | CVR-AUC Δ | CTCVR-AUC Δ |
|-------|-----------|-------------|
| ESCM2_IPW | +0.0219 | +0.0298 |
| DCMT | +0.0187 | +0.0248 |
| DDPO | +0.0158 | +0.0160 |
| CHORUS | +0.0181 | +0.0173 |
