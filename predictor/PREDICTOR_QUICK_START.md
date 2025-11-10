# Predictor Quick Start File

## Start Service

```
uv sync

uv run uvicorn src.api:app --host <your_host> --port <your_port>
```

## Train & Inference Model

> Reference: `predictor/docs/5.API_ENDPOINTS.md`
> Datamodel Refence: `predictor/docs/4.DATA_MODELS.md`

After training, model will be save to `models`

## 使用说明

训练：  
1. API使用 `model_id` 和 `platform_info` 区分不同的模型和目标平台，使用 `features_list` 训练模型
2. `prediction_type` 指定预测类型，`quantile`即为分位数预测模式
3. 默认MLP参数未进行调优，需要通过`training_config`配置MLP以及目标预测分位数
4. `features_list`中`runtime_ms`将会被作为预测目标，其他参数将会被作为特征

预测：
1. `features`内部格式与`features_list`中单个元素一致
2. 需要使用和训练时一致的`model_id`和`platofmr_info`识别预测用模型
3. 若`features`中出现`exp_runtime`，将会启用假数据bypass路径，不会经过模型进行预测
4. 预测分位数设置取决于训练时的分位数设置，预测时无法调整目标的分位数设置