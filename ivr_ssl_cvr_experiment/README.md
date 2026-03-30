# IVR SSL CVR Project

Self-Supervised Learning for CVR Prediction on IVR Dataset, aimed at addressing negative sampling imbalance problems. Features are aligned with the multitask project's `combine_schema` for consistency.

## Project Structure
- `src/`: Core model code
- `scripts/`: Experiment scripts
- `results/`: Experiment results
- `data/`: Data preprocessing cache

## Dataset
- **Source**: `/mnt/data/oss_dsp_algo/ivr/sample/ivr_sample_v16/`
- **Time Range**: 2025-11-01 to 2026-03-09
- **Features**: 125 features from combine_schema + behavioral labels (click, purchase, atc, cv)
- **Business Types**: shein, aecps, aedsp, aerta, shopee_cps, lazada_cps, lazada_rta
- **Challenge**: Severe negative sampling imbalance (known issue with IVR data)

## Research Goals
1. Apply contrastive learning techniques to improve CVR prediction
2. Address negative sampling imbalance in IVR data
3. Explore SSL methods suitable for IVR scenarios
4. Leverage cross-business similarities for better generalization
5. Ensure feature consistency with multitask project through combine_schema

## Implemented Components
1. **Schema-Compliant Data Loader**: Reads IVR parquet files with proper mul_labels parsing
2. **Contrastive Models**: SimCLR, Supervised Contrastive, Business-Type Contrastive
3. **Loss Functions**: InfoNCE and Supervised Contrastive Loss implementations
4. **Feature Processing**: Handles categorical features from combine_schema

## Quick Start
```bash
# Run component tests
python test_components.py

# Test data loader with schema-compliant features
python src/data_loader_simple.py

# Run dataset analysis
python analyze_dataset.py

# Run experiments
python run_experiments.py
```

## Key Features
- **combine_schema Compliance**: Uses exact features from multitask project
- **mul_labels Parsing**: Properly handles list-of-tuples format in mul_labels column
- **Business Type Filtering**: Focuses on target business types (shein, aecps, etc.)
- **Focused Feature Set**: Concentrates on available features like business_type for contrastive learning
- **Memory Efficient**: Proper sampling and batch processing