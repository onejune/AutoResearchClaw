# Experiment 14: TabTransformer Results

## Paper Reference
- **Title**: TabTransformer: Tabular Data Modeling Using Contextual Embeddings
- **Venue**: KDD 2020

## Method
- Self-attention for feature interaction
- Transformer encoder architecture
- Fusion of continuous and categorical features

## Configuration
- **Samples**: 100,000
- **d_model**: 64
- **n_heads**: 4
- **n_layers**: 2

## Results

### Win Rate Prediction
| Metric | Value |
|--------|-------|
| AUC | 0.8432 |
| RMSE | 0.5048 |
| MAE | 0.2548 |
| ECE | 0.0110 |

### Training Info
- **Device**: cuda
- **Parameters**: 69,313
- **Training Time**: 65.84s

---
*Generated: 2026-04-01 17:35:33*
