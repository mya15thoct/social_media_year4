# Fraud Detection in Supply Chain

Dự án phát hiện giao dịch gian lận trong chuỗi cung ứng sử dụng Deep Learning kết hợp với Social Network Analysis.

## Tóm tắt

Mô hình DNN ensemble kết hợp features từ giao dịch (57 features) và network topology (8 features) để detect fraud trong supply chain dataset. Model đạt **82.54% ROC-AUC** và **75.17% Recall**.

**Dataset**: DataCo Supply Chain (180K orders, 20K customers, fraud rate: 6.9%)

## Cấu trúc thư mục

```
data/
├── raw/          # Dataset gốc (96 MB)
├── processed/    # Features đã xử lý (35 MB)
└── intermediate/ # Graph files, temp files (48 MB)

Fraud_SupplyChain/
├── analysis/     # Ablation study, model comparison
├── model/        # DNN model, training code
├── clean_data.py
├── extract_transaction_features.py
└── merge_features.py

SNA/              # Network analysis scripts
```

## Setup

```bash
pip install -r requirements.txt
```

## Cách chạy

### 1. Extract features từ dataset gốc

```bash
cd Fraud_SupplyChain
python extract_transaction_features.py
```

Output: `data/processed/transaction_features.csv`

### 2. Merge transaction + network features

```bash
python merge_features.py
```

Output: 
- `data/processed/combined_features.csv`
- `data/processed/transaction_only.csv`
- `data/processed/network_only.csv`

### 3. Train model

```bash
cd model
python main_ensemble.py
```

Model sẽ train ensemble 3 models với seeds khác nhau, results lưu ở `model/results/`

### 4. Analysis

```bash
cd analysis

# Ablation study - test các feature sets khác nhau
python ablation_study_dnn.py

# So sánh với ML models khác (RF, XGBoost, LightGBM)
python alternative_models.py

# Visualize network
python visualize_bipartite_network.py
```

## Kết quả chính

| Model | Features | AUC | Recall | Precision |
|-------|----------|-----|--------|-----------|
| DNN Ensemble | Transaction + Network (65) | **82.54%** | **75.17%** | 18.86% |
| DNN Ensemble | Transaction only (57) | 82.16% | 74.83% | 20.15% |

Network features giúp tăng AUC thêm **+0.38%** và catch thêm được 1 fraud case.

## Model config

Xem chi tiết tại `Fraud_SupplyChain/model/config.py`:
- Architecture: 256 → 128 → 64 với BatchNorm + Dropout
- Loss: Cost-Sensitive Focal Loss (FN_cost = 15x)
- SMOTE: Balanced sampling (1.0)
- Threshold: 0.20 (optimized cho high recall)

## Network Features

Từ bipartite graph Customer-Product (20K customers, 118 products, 101K edges):
- Degree/Betweenness/Closeness Centrality
- PageRank
- Eigenvector Centrality
- Clustering Coefficient
- Community ID

## Notes

Nếu gặp lỗi corrupted data, chạy:
```bash
python clean_data.py
```

## Author

Research project cho môn Social Network Analysis & Fraud Detection
