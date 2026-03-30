# data_v3 实验结果汇总

生成时间：2026-03-29 20:32:26

数据集：data_v3（训练集 2,728,570 行，测试集 992,085 行）

## Overall AUC 对比

| 模型 | Overall AUC | aecps | aedsp | aerta | lazada_cps | lazada_rta | shein | shopee_cps |
|------|-------------|------|------|------|------|------|------|------|
| baseline | 0.4883 | 0.5343 | 0.5595 | 0.5359 | 0.5153 | 0.5179 | 0.471 | 0.5039 |
| mlora_rank8 | 0.4935 | 0.527 | 0.5539 | 0.5391 | 0.514 | 0.514 | 0.4994 | 0.5057 |
| mlora_rank12 | 0.4890 | 0.5517 | 0.5777 | 0.5615 | 0.5262 | 0.5108 | 0.485 | 0.5075 |
| user_contrastive | 0.4944 | 0.5292 | 0.5482 | 0.5406 | 0.5168 | 0.5119 | 0.4985 | 0.5065 |
| mlora_rank8_user_cl0.1 | 0.4940 | 0.5285 | 0.5555 | 0.5269 | 0.515 | 0.5182 | 0.505 | 0.5074 |
| mlora_rank12_user_cl0.1 | 0.4935 | 0.5358 | 0.5429 | 0.535 | 0.5138 | 0.5164 | 0.503 | 0.5063 |

## Per-BT AUC 详情

### baseline
- Overall AUC: **0.4883**
- cl_weight: 0.0

| business_type | AUC |
|---------------|-----|
| aecps | 0.5343 |
| aedsp | 0.5595 |
| aerta | 0.5359 |
| lazada_cps | 0.5153 |
| lazada_rta | 0.5179 |
| shein | 0.471 |
| shopee_cps | 0.5039 |

### mlora_rank8
- Overall AUC: **0.4935**
- cl_weight: 0.0

| business_type | AUC |
|---------------|-----|
| aecps | 0.527 |
| aedsp | 0.5539 |
| aerta | 0.5391 |
| lazada_cps | 0.514 |
| lazada_rta | 0.514 |
| shein | 0.4994 |
| shopee_cps | 0.5057 |

### mlora_rank12
- Overall AUC: **0.4890**
- cl_weight: 0.0

| business_type | AUC |
|---------------|-----|
| aecps | 0.5517 |
| aedsp | 0.5777 |
| aerta | 0.5615 |
| lazada_cps | 0.5262 |
| lazada_rta | 0.5108 |
| shein | 0.485 |
| shopee_cps | 0.5075 |

### user_contrastive
- Overall AUC: **0.4944**
- cl_weight: 0.1

| business_type | AUC |
|---------------|-----|
| aecps | 0.5292 |
| aedsp | 0.5482 |
| aerta | 0.5406 |
| lazada_cps | 0.5168 |
| lazada_rta | 0.5119 |
| shein | 0.4985 |
| shopee_cps | 0.5065 |

### mlora_rank8_user_cl0.1
- Overall AUC: **0.4940**
- cl_weight: 0.1

| business_type | AUC |
|---------------|-----|
| aecps | 0.5285 |
| aedsp | 0.5555 |
| aerta | 0.5269 |
| lazada_cps | 0.515 |
| lazada_rta | 0.5182 |
| shein | 0.505 |
| shopee_cps | 0.5074 |

### mlora_rank12_user_cl0.1
- Overall AUC: **0.4935**
- cl_weight: 0.1

| business_type | AUC |
|---------------|-----|
| aecps | 0.5358 |
| aedsp | 0.5429 |
| aerta | 0.535 |
| lazada_cps | 0.5138 |
| lazada_rta | 0.5164 |
| shein | 0.503 |
| shopee_cps | 0.5063 |
