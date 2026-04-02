# Experiment 12: DeepWin Results

## Paper Reference
- **Title**: DeepWin: A Deep Recurrent Model for Real-Time Auction Win Rate Prediction
- **Venue**: ACM TIST, 2026

## Configuration
- **Samples**: 100,000
- **Bid Sequence Length**: 5
- **LSTM Hidden**: 128
- **LSTM Layers**: 2
- **Bid Embedding Dim**: 16
- **Batch Size**: 256
- **Learning Rate**: 0.001
- **Epochs**: 50 (with early stopping)

## Results

### Win Rate Prediction
| Metric | Value |
|--------|-------|
| AUC | 0.9497 |
| RMSE | 0.3516 |
| MAE | 0.1236 |
| ECE | 0.0065 |
| PCOC | 0.7033 |
| Brier Score | 0.0876 |

### Training Info
- **Device**: cuda
- **Total Parameters**: 231,538
- **Training Time**: 29.33s

## Comparison with exp05 (DLF-GRU)
| Model | AUC | RMSE | ECE | PCOC |
|-------|-----|------|-----|------|
| exp05 (DLF-GRU) | 0.8687 | 0.3841 | 0.0052 | - |
| **exp12 (DeepWin)** | 0.9497 | 0.3516 | 0.0065 | 0.7033 |

## Key Insights
1. LSTM + Attention vs GRU: Improved performance
2. Attention mechanism provides interpretability
3. Bid embedding captures non-linear price relationships

---
*Generated: 2026-04-01 12:42:45*
