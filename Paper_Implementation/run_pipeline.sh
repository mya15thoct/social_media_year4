#!/bin/bash
# Script to run Paper Implementation on server

echo "=========================================="
echo "Paper Implementation - IJISA-V17-N2-2"
echo "=========================================="

# Step 1: Install dependencies
echo ""
echo "[1/4] Installing dependencies..."
pip install -r requirements.txt

# Step 2: Run preprocessing
echo ""
echo "[2/4] Running preprocessing (PCA + SMOTE)..."
python preprocessing.py

# Step 3: Train model
echo ""
echo "[3/4] Training model (50 epochs)..."
python train.py

# Step 4: Evaluate model
echo ""
echo "[4/4] Evaluating model..."
python evaluate.py

echo ""
echo "=========================================="
echo "✅ Pipeline complete!"
echo "=========================================="
echo "Results saved to:"
echo "  - models/saved_models/best_model.keras"
echo "  - results/figures/"
echo "  - results/metrics/"
