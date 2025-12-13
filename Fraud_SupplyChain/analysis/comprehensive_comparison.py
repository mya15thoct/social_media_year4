"""
COMPREHENSIVE MODEL COMPARISON SCRIPT
Compare multiple models and feature sets to evaluate impact of network features
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
import os
import sys

print("="*80)
print("COMPREHENSIVE MODEL COMPARISON")
print("="*80)

# Load data
print("\n[1] Loading data...")
script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(script_dir, '..', 'data')

df_combined = pd.read_csv(os.path.join(data_dir, 'combined_features.csv'))
df_transaction = pd.read_csv(os.path.join(data_dir, 'transaction_only.csv'))
df_network = pd.read_csv(os.path.join(data_dir, 'network_only.csv'))

print(f"Combined: {df_combined.shape}")
print(f"Transaction only: {df_transaction.shape}")
print(f"Network only: {df_network.shape}")

# Define feature sets for comparison
feature_sets = {
    '1_Transaction_Only': {
        'data': df_transaction,
        'description': 'Baseline: Transaction features only (57 features)'
    },
    '2_Transaction_OldNetwork': {
        'data': df_combined[[c for c in df_combined.columns if c not in [
            'eigenvector_centrality', 'clustering_coefficient', 
            'num_products', 'avg_product_popularity'
        ]]],
        'description': 'Transaction + Old Network (61 features: 57+4)'
    },
    '3_Transaction_NewNetwork': {
        'data': df_combined,
        'description': 'Transaction + New Network (65 features: 57+8)'
    },
    '4_Network_Only': {
        'data': df_network,
        'description': 'Network features only (8 features)'
    }
}

# Results storage
results = []

# Train and evaluate each feature set
print("\n" + "="*80)
print("TRAINING MODELS")
print("="*80)

for name, config in feature_sets.items():
    print(f"\n{'='*80}")
    print(f"Feature Set: {name}")
    print(f"Description: {config['description']}")
    print(f"{'='*80}")
    
    # Prepare data
    df = config['data']
    X = df.drop(['Customer Id', 'is_fraud'], axis=1, errors='ignore')
    y = df['is_fraud']
    
    print(f"Features: {X.shape[1]}")
    print(f"Samples: {len(X):,}")
    print(f"Fraud rate: {y.mean():.2%}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train Random Forest (fast and interpretable)
    print("\nTraining Random Forest...")
    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=20,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_train_scaled, y_train)
    
    # Predict
    y_pred_proba = rf.predict_proba(X_test_scaled)[:, 1]
    
    # Try different thresholds
    thresholds = [0.2, 0.3, 0.4, 0.5]
    best_f1 = 0
    best_threshold = 0.5
    best_metrics = {}
    
    for threshold in thresholds:
        y_pred = (y_pred_proba > threshold).astype(int)
        f1 = f1_score(y_test, y_pred)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
            best_metrics = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred, zero_division=0),
                'recall': recall_score(y_test, y_pred),
                'f1': f1,
                'auc': roc_auc_score(y_test, y_pred_proba),
                'cm': confusion_matrix(y_test, y_pred)
            }
    
    # Store results
    results.append({
        'name': name,
        'description': config['description'],
        'n_features': X.shape[1],
        'threshold': best_threshold,
        **best_metrics
    })
    
    # Print results
    print(f"\nBest threshold: {best_threshold}")
    print(f"Accuracy:  {best_metrics['accuracy']:.4f}")
    print(f"Precision: {best_metrics['precision']:.4f}")
    print(f"Recall:    {best_metrics['recall']:.4f}")
    print(f"F1-Score:  {best_metrics['f1']:.4f}")
    print(f"ROC-AUC:   {best_metrics['auc']:.4f}")
    print(f"\nConfusion Matrix:")
    print(best_metrics['cm'])

# Create comparison table
print("\n" + "="*80)
print("COMPARISON SUMMARY")
print("="*80)

df_results = pd.DataFrame(results)
print("\n" + df_results.to_string(index=False))

# Save results
os.makedirs('results', exist_ok=True)
df_results.to_csv('results/model_comparison.csv', index=False)
print(f"\n✓ Saved to: results/model_comparison.csv")

# Feature importance for best model (Transaction + New Network)
print("\n" + "="*80)
print("FEATURE IMPORTANCE (Transaction + New Network)")
print("="*80)

X_full = df_combined.drop(['Customer Id', 'is_fraud'], axis=1)
y_full = df_combined['is_fraud']
X_train_full, X_test_full, y_train_full, y_test_full = train_test_split(
    X_full, y_full, test_size=0.2, random_state=42, stratify=y_full
)

scaler_full = StandardScaler()
X_train_full_scaled = scaler_full.fit_transform(X_train_full)

rf_full = RandomForestClassifier(
    n_estimators=100, max_depth=10, class_weight='balanced', 
    random_state=42, n_jobs=-1
)
rf_full.fit(X_train_full_scaled, y_train_full)

# Get feature importance
importance_df = pd.DataFrame({
    'feature': X_full.columns,
    'importance': rf_full.feature_importances_
}).sort_values('importance', ascending=False)

print("\nTop 20 Most Important Features:")
print(importance_df.head(20).to_string(index=False))

# Save feature importance
importance_df.to_csv('results/feature_importance.csv', index=False)
print(f"\n✓ Saved to: results/feature_importance.csv")

# Network features importance
network_features = [c for c in X_full.columns if any(x in c.lower() for x in 
    ['centrality', 'pagerank', 'cluster', 'community', 'degree'])]
network_importance = importance_df[importance_df['feature'].isin(network_features)]

print(f"\n{'='*80}")
print("NETWORK FEATURES IMPORTANCE")
print(f"{'='*80}")
print(network_importance.to_string(index=False))

print("\n" + "="*80)
print("ANALYSIS COMPLETE!")
print("="*80)
