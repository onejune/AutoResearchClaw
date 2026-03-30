# IVR SSL CVR Project - Updated Summary

## Project Overview
This project applies contrastive learning techniques to improve CVR (Conversion Rate) prediction on the IVR dataset, specifically addressing the known negative sampling imbalance problem. The implementation now uses features from the `combine_schema` to ensure consistency with the multitask project.

## Dataset Information
- **Source**: `/mnt/data/oss_dsp_algo/ivr/sample/ivr_sample_v16/`
- **Time Range**: 2025-11-01 to 2026-03-09
- **Business Types**: shein, aecps, aedsp, aerta, shopee_cps, lazada_cps, lazada_rta
- **Features**: 125 features from combine_schema plus behavioral labels

## Behavioral Labels (parsed from mul_labels)
- **click**: Click signal (binary)
- **purchase**: Purchase signal (binary) 
- **atc**: Add-to-cart signal (binary)
- **cv**: Conversion signal (binary)

## Schema-Compliant Features
The model uses features from the multitask project's `combine_schema` including:
- `business_type`: Business type categorization
- `expid`: Experiment ID
- `region`: Geographic region
- `imptype`: Impression type
- `objective_type`: Campaign objective
- `traffic_type`: Traffic type

## Key Improvements
1. **Schema Compliance**: All features now align with `combine_schema` from multitask project
2. **Proper Data Loading**: Fixed issues with PyArrow map types in `mul_labels` column
3. **Robust Parsing**: Correctly handles list-of-tuples format in `mul_labels`
4. **Memory Efficiency**: Implements proper sampling and filtering

## Contrastive Learning Approaches
The project supports multiple contrastive learning strategies:
1. **SimCLR-style**: Unsupervised contrastive learning
2. **Supervised Contrastive**: Using purchase labels for positive pairs
3. **Business-Type Contrastive**: Contrasting across business types
4. **Temporal Contrastive**: Time-based contrastive learning

## Next Steps
1. Scale up experiments with larger datasets
2. Implement advanced augmentation strategies
3. Fine-tune hyperparameters for optimal performance
4. Compare against baseline models from multitask project