"""
ABLATION STUDY - DNN MODEL (PROPER IMPLEMENTATION)
===================================================
This script performs ablation study using the MAIN DNN model architecture.

WHY THIS IS NEEDED:
- Previous ablation in comprehensive_comparison.py used Random Forest
- Ablation study MUST use the main model (DNN) to measure feature impact
- This gives accurate results for the DNN model, not RF

Feature Sets Tested:
1. Transaction Only (57 features)
2. Transaction + Old Network (61 features: 57+4)
3. Transaction + New Network (65 features: 57+8) ← Main model
4. Network Only (8 features)
"""

import pandas as pd
import numpy as np
import os
import sys
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)

# Import from main model
script_dir = os.path.dirname(os.path.abspath(__file__))
model_dir = os.path.join(script_dir, '..', 'model')
sys.path.insert(0, model_dir)

import config
import data_loader
import model as dnn_model
import predict

print("="*80)
print("ABLATION STUDY - DNN MODEL")
print("="*80)
print("\n⚠️  This ablation study uses the MAIN DNN architecture")
print("   - Architecture: Dense(256) → Dense(128) → Dense(64)")
print("   - Loss: Cost-Sensitive Focal Loss")
print("   - Ensemble: 3 models (seeds: 42, 123, 456)")
print("="*80)

# Load data files
data_dir = os.path.join(script_dir, '..', 'data')
df_combined = pd.read_csv(os.path.join(data_dir, 'combined_features.csv'))
df_transaction = pd.read_csv(os.path.join(data_dir, 'transaction_only.csv'))
df_network = pd.read_csv(os.path.join(data_dir, 'network_only.csv'))

print(f"\n[Data Loaded]")
print(f"Combined: {df_combined.shape}")
print(f"Transaction only: {df_transaction.shape}")
print(f"Network only: {df_network.shape}")

# Define feature sets
feature_sets = {
    '1_Transaction_Only': {
        'data': df_transaction,
        'description': 'Baseline: Transaction features only (57 features)',
        'n_features': 57
    },
    '2_Transaction_OldNetwork': {
        'data': df_combined[[c for c in df_combined.columns if c not in [
            'eigenvector_centrality', 'clustering_coefficient', 
            'num_products', 'avg_product_popularity'
        ]]],
        'description': 'Transaction + Old Network (61 features: 57+4)',
        'n_features': 61
    },
    '3_Transaction_NewNetwork': {
        'data': df_combined,
        'description': 'Transaction + New Network (65 features: 57+8) ← MAIN MODEL',
        'n_features': 65
    },
    '4_Network_Only': {
        'data': df_network,
        'description': 'Network features only (8 features)',
        'n_features': 8
    }
}

# Results storage
results = []

# Train each feature set with DNN
print("\n" + "="*80)
print("TRAINING DNN ON DIFFERENT FEATURE SETS")
print("="*80)

for name, feature_config in feature_sets.items():
    print(f"\n{'='*80}")
    print(f"Feature Set: {name}")
    print(f"Description: {feature_config['description']}")
    print(f"Features: {feature_config['n_features']}")
    print(f"{'='*80}")
    
    # Prepare data
    df = feature_config['data']
    X = df.drop(['Customer Id', 'is_fraud'], axis=1, errors='ignore')
    y = df['is_fraud']
    
    print(f"Samples: {len(X):,}")
    print(f"Fraud rate: {y.mean():.2%}")
    
    # Split data (same as main model)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale
    X_train_scaled, X_test_scaled, scaler = data_loader.scale_data(
        X_train.values, X_test.values
    )
    
    # Ensemble predictions storage
    ensemble_predictions = []
    
    # Train ensemble (3 models like main)
    for i, seed in enumerate(config.ENSEMBLE_SEEDS, 1):
        print(f"\n  Training model {i}/{len(config.ENSEMBLE_SEEDS)} (seed={seed})...")
        
        # Apply SMOTE
        X_train_res, y_train_res = data_loader.apply_smote(
            X_train_scaled, y_train, 
            random_state=seed,
            sampling_strategy=config.SAMPLING_STRATEGY
        )
        
        # Clean NaN/Inf
        X_train_res = np.nan_to_num(X_train_res, nan=0.0, posinf=0.0, neginf=0.0)
        X_test_scaled_clean = np.nan_to_num(X_test_scaled, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Apply PCA
        pca = PCA(n_components=min(config.N_COMPONENTS, X_train_res.shape[1]), random_state=seed)
        X_train_pca = pca.fit_transform(X_train_res)
        X_test_pca = pca.transform(X_test_scaled_clean)
        
        explained_var = pca.explained_variance_ratio_.sum()
        print(f"    PCA: {X_train_res.shape[1]} → {X_train_pca.shape[1]} components ({explained_var*100:.1f}% variance)")
        
        # Split train/validation
        X_train_final, X_val, y_train_final, y_val = train_test_split(
            X_train_pca, y_train_res,
            test_size=config.VALIDATION_SPLIT,
            random_state=seed
        )
        
        # Build DNN model
        input_dim = X_train_pca.shape[1]
        fraud_model = dnn_model.build_model(
            input_dim, 
            use_focal_loss=config.USE_FOCAL_LOSS,
            focal_gamma=config.FOCAL_GAMMA,
            focal_alpha=config.FOCAL_ALPHA,
            use_cost_sensitive=config.USE_COST_SENSITIVE,
            fn_cost=config.FN_COST
        )
        
        # Train (silently)
        fraud_model.fit(
            X_train_final, y_train_final,
            validation_data=(X_val, y_val),
            epochs=config.EPOCHS,
            batch_size=config.BATCH_SIZE,
            verbose=0,
            callbacks=[
                __import__('tensorflow').keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=10,
                    restore_best_weights=True
                )
            ]
        )
        
        # Predict
        y_pred_proba = fraud_model.predict(X_test_pca, verbose=0).flatten()
        ensemble_predictions.append(y_pred_proba)
        
        print(f"    ✓ Model {i} trained")
    
    # Average ensemble predictions
    y_pred_proba_ensemble = np.mean(ensemble_predictions, axis=0)
    
    # Find optimal threshold
    if config.THRESHOLD == 'auto':
        threshold = predict.find_optimal_threshold(y_test, y_pred_proba_ensemble, metric='balanced')
    else:
        threshold = config.THRESHOLD
    
    # Make final predictions
    y_pred = (y_pred_proba_ensemble > threshold).astype(int)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_test, y_pred_proba_ensemble)
    cm = confusion_matrix(y_test, y_pred)
    
    # Store results
    results.append({
        'Feature_Set': name,
        'Description': feature_config['description'],
        'N_Features': feature_config['n_features'],
        'Threshold': threshold,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1_Score': f1,
        'ROC_AUC': roc_auc,
        'TN': cm[0, 0],
        'FP': cm[0, 1],
        'FN': cm[1, 0],
        'TP': cm[1, 1]
    })
    
    # Print results
    print(f"\n  {'='*60}")
    print(f"  RESULTS")
    print(f"  {'='*60}")
    print(f"  Threshold: {threshold:.3f}")
    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1-Score:  {f1:.4f}")
    print(f"  ROC-AUC:   {roc_auc:.4f}")
    print(f"\n  Confusion Matrix:")
    print(f"  {cm}")

# Create comparison table
print("\n" + "="*80)
print("ABLATION STUDY RESULTS - DNN MODEL")
print("="*80)

df_results = pd.DataFrame(results)

# Format for display
df_display = df_results[['Feature_Set', 'N_Features', 'ROC_AUC', 'Recall', 'Precision', 'F1_Score']].copy()
df_display['ROC_AUC'] = df_display['ROC_AUC'].apply(lambda x: f"{x:.4f}")
df_display['Recall'] = df_display['Recall'].apply(lambda x: f"{x:.4f}")
df_display['Precision'] = df_display['Precision'].apply(lambda x: f"{x:.4f}")
df_display['F1_Score'] = df_display['F1_Score'].apply(lambda x: f"{x:.4f}")

print("\n" + df_display.to_string(index=False))

# Calculate improvements
baseline_idx = 1  # Transaction + Old Network (61 features)
enhanced_idx = 2  # Transaction + New Network (65 features)

baseline = results[baseline_idx]
enhanced = results[enhanced_idx]

print("\n" + "="*80)
print("FEATURE IMPACT ANALYSIS")
print("="*80)
print(f"\nBaseline (61 features):  AUC={baseline['ROC_AUC']:.4f}, Recall={baseline['Recall']:.4f}")
print(f"Enhanced (65 features):  AUC={enhanced['ROC_AUC']:.4f}, Recall={enhanced['Recall']:.4f}")
print(f"\nImprovement:")
print(f"  ΔR AUC:    {(enhanced['ROC_AUC'] - baseline['ROC_AUC'])*100:+.2f}%")
print(f"  Δ Recall:  {(enhanced['Recall'] - baseline['Recall'])*100:+.2f}%")
print(f"  Δ TP:      {enhanced['TP'] - baseline['TP']:+d} frauds caught")
print(f"  Δ FN:      {enhanced['FN'] - baseline['FN']:+d} frauds missed")

# Save results
results_dir = os.path.join(script_dir, 'results')
os.makedirs(results_dir, exist_ok=True)

csv_path = os.path.join(results_dir, 'ablation_study_dnn_results.csv')
df_results.to_csv(csv_path, index=False)

txt_path = os.path.join(results_dir, 'ablation_study_dnn_summary.txt')
with open(txt_path, 'w') as f:
    f.write("="*80 + "\n")
    f.write("ABLATION STUDY - DNN MODEL\n")
    f.write("="*80 + "\n\n")
    f.write(df_display.to_string(index=False))
    f.write("\n\n" + "="*80 + "\n")
    f.write("FEATURE IMPACT ANALYSIS\n")
    f.write("="*80 + "\n")
    f.write(f"\nBaseline (61 features):  AUC={baseline['ROC_AUC']:.4f}, Recall={baseline['Recall']:.4f}\n")
    f.write(f"Enhanced (65 features):  AUC={enhanced['ROC_AUC']:.4f}, Recall={enhanced['Recall']:.4f}\n")
    f.write(f"\nImprovement:\n")
    f.write(f"  Δ AUC:    {(enhanced['ROC_AUC'] - baseline['ROC_AUC'])*100:+.2f}%\n")
    f.write(f"  Δ Recall:  {(enhanced['Recall'] - baseline['Recall'])*100:+.2f}%\n")
    f.write(f"  Δ TP:      {enhanced['TP'] - baseline['TP']:+d} frauds caught\n")
    f.write(f"  Δ FN:      {enhanced['FN'] - baseline['FN']:+d} frauds missed\n")

print(f"\n✓ Results saved:")
print(f"  - {csv_path}")
print(f"  - {txt_path}")

print("\n" + "="*80)
print("✓ ABLATION STUDY COMPLETE!")
print("="*80)
