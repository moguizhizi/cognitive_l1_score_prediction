```text
cognitive_l1_score_prediction/
│
├── README.md
├── requirements.txt
├── setup.py
├── .gitignore
│
├── configs/                       # 配置文件目录
│   ├── __init__.py
│   ├── loader.py
│   └── train.yaml
│
├── data/                           # 数据目录（不提交大文件）
│   ├── raw/                        # 原始数据
│   ├── processed/                  # 清洗后数据
│   └── features/                   # 特征工程结果
│
├── notebooks/                      # 实验分析 notebook
│   ├── EDA.ipynb
│   ├── feature_analysis.ipynb
│   └── error_analysis.ipynb
│
├── src/                            # 核心源码
│   │
│   ├── __init__.py
│   │
│   ├── core/                       # 项目核心定义：全局常量、枚举、数据Schema等
│   │   ├── __init__.py
│   │   └── constants.py
│   │
│   ├── data/                       # 数据处理模块
│   │   ├── loader.py
│   │   ├── preprocess.py
│   │   ├── splitter.py
│   │   │
│   │   ├── split/                  # 数据集划分模块
│   │   │   ├── __init__.py
│   │   │   └── cognitive_l1_splitter.py
│   │   │
│   │   ├── analysis/               # 数据分析（EDA）模块
│   │   │   ├── __init__.py
│   │   │   └── eda_cognitive_l1_dataset.py
│   │   │
│   │   └── datasets/
│   │       ├── __init__.py
│   │       └── cognitive_l1_dataset.py
│   │
│   ├── features/                   # 特征工程
│   │   ├── feature_builder.py
│   │   ├── feature_selector.py
│   │   ├── time_series_features.py
│   │   └── feature_utils.py
│   │
│   ├── models/                     # 模型定义
│   │   ├── base_model.py
│   │   ├── xgboost_model.py
│   │   ├── lightgbm_model.py
│   │   ├── least_square_model.py
│   │   ├── mlp_model.py
│   │   └── model_factory.py
│   │
│   ├── training/                   # 训练流程
│   │   ├── trainer.py
│   │   ├── loss.py
│   │   ├── optimizer.py
│   │   └── scheduler.py
│   │
│   ├── evaluation/                 # 评估模块
│   │   ├── metrics.py
│   │   ├── evaluator.py
│   │   └── visualization.py
│   │
│   ├── inference/                  # 推理核心逻辑
│   │   ├── predictor.py
│   │   └── batch_infer.py
│   │
│   ├── utils/                      # 通用工具
│   │   ├── logger.py
│   │   ├── seed.py
│   │   ├── io_utils.py
│   │   ├── json_utils.py
│   │   ├── parquet_utils.py
│   │   ├── text_utils.py
│   │   ├── xlsx_utils.py
│   │   ├── path_utils.py
│   │   ├── dataframe_utils.py
│   │   └── config_loader.py
│   │
│   └── pipelines/                  # 任务级 pipeline
│       ├── build_dataset_pipeline.py
│       ├── __init__.py
│       │
│       ├── train_pipleline/        # 训练 pipeline
│       │   ├── cognitive_l1.py
│       │   └── train_cognitive_l1_models.py
│       │
│       └── infer_pipleline/        # 推理 pipeline
│           └── infer_cognitive_l1_models.py
│
├── scripts/                        # 命令行脚本
│   ├── train.sh
│   ├── infer.sh
│   └── evaluate.sh
│
├── experiments/                    # 实验记录
│   ├── exp_001_baseline/
│   ├── exp_002_feature_v2/
│   └── logs/
│
├── checkpoints/                    # 模型权重
│   ├── best_model.pt
│   └── last_model.pt
│
└── tests/                          # 单元测试
    ├── test_data.py
    ├── test_model.py
    └── test_metrics.py