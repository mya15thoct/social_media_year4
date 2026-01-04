# Fraud Detection in Supply Chain

Deep Learning + Social Network Analysis for fraud detection in supply chain transactions.

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run Complete Pipeline

```bash
# Step 1: Extract transaction features
cd Fraud_SupplyChain/preprocessing
python extract_transaction_features.py

# Step 2: Build network and calculate features
cd ../network
python create_edgelist.py
python build_network.py
python calculate_network_features.py

# Step 3: Merge all features
cd ../preprocessing
python merge_features.py

# Step 4: Train model
cd ../model
python main_ensemble.py

# Step 5: Run analysis
cd ../analysis
python comprehensive_comparison.py
```

## Project Structure

```
Fraud_SupplyChain/
├── preprocessing/     # Data cleaning & feature extraction
├── network/          # Social network analysis
├── model/            # DNN training
└── analysis/         # Model evaluation
```



## Configuration

Edit `Fraud_SupplyChain/model/config.py` to adjust:
- Model architecture (layers, dropout)
- Training parameters (epochs, batch size)
- Loss function (cost-sensitive focal loss)
- Threshold for classification

## Troubleshooting

**Corrupted data:**
```bash
cd Fraud_SupplyChain/preprocessing
python clean_data.py
```

**Missing dependencies:**
```bash
pip install -r requirements.txt --upgrade
```

## Requirements

- Python 3.8+
- TensorFlow 2.12+
- NetworkX 3.0+
- See `requirements.txt` for full list

## Dataset

Place `DataCoSupplyChainDataset.csv` in `data/raw/` directory.

