"""
Merge Transaction Features and Network Features
Creates combined dataset for model comparison
"""

import pandas as pd
import os

def load_transaction_features(file_path):
    """Load transaction features"""
    print(f"Loading transaction features from {file_path}...")
    df = pd.read_csv(file_path)
    print(f"  Loaded {len(df)} customers with {df.shape[1]} columns")
    return df

def load_network_features(file_path):
    """Load network features"""
    print(f"\nLoading network features from {file_path}...")
    df = pd.read_csv(file_path)
    print(f"  Loaded {len(df)} customers with {df.shape[1]} columns")
    return df

def merge_features(df_transaction, df_network):
    """Merge transaction and network features on Customer Id"""
    print("\nMerging features on Customer Id...")
    
    # Rename customer_id in network features to match transaction features
    if 'customer_id' in df_network.columns:
        df_network.rename(columns={'customer_id': 'Customer Id'}, inplace=True)
    
    # Select only network features (exclude is_fraud from network_features)
    network_cols = ['Customer Id', 'degree_centrality', 'betweenness_centrality', 
                    'closeness_centrality', 'pagerank', 'eigenvector_centrality',
                    'clustering_coefficient', 'community_id']

    
    # Check which columns exist
    existing_network_cols = [col for col in network_cols if col in df_network.columns]
    df_network_selected = df_network[existing_network_cols].copy()
    
    # Merge
    df_merged = pd.merge(df_transaction, df_network_selected, 
                         on='Customer Id', how='inner')
    
    print(f"  Merged dataset: {len(df_merged)} customers")
    print(f"  Total features: {df_merged.shape[1] - 2} (excluding Customer Id and is_fraud)")
    
    # Count features
    transaction_features = df_transaction.shape[1] - 2  # exclude Customer Id and is_fraud
    network_features = len(existing_network_cols) - 1   # exclude Customer Id
    
    print(f"\nFeature breakdown:")
    print(f"  Transaction features: {transaction_features}")
    print(f"  Network features: {network_features}")
    print(f"  Combined features: {transaction_features + network_features}")
    
    return df_merged

def save_separate_datasets(df_merged):
    """Save 3 versions: transaction-only, network-only, combined"""
    
    # Identify feature columns
    exclude_cols = ['Customer Id', 'is_fraud']
    all_features = [col for col in df_merged.columns if col not in exclude_cols]
    
    network_features = ['degree_centrality', 'betweenness_centrality', 
                       'closeness_centrality', 'pagerank', 'eigenvector_centrality',
                       'clustering_coefficient', 'community_id']

    
    # Filter network features that exist
    network_features = [col for col in network_features if col in df_merged.columns]
    
    # Transaction features = all features - network features
    transaction_features = [col for col in all_features if col not in network_features]
    
    print("\nSaving datasets...")
    
    # 1. Transaction-only
    df_transaction_only = df_merged[['Customer Id'] + transaction_features + ['is_fraud']].copy()
    df_transaction_only.to_csv('data/transaction_only.csv', index=False)
    print(f"  ✅ Transaction-only: {len(transaction_features)} features → data/transaction_only.csv")
    
    # 2. Network-only
    df_network_only = df_merged[['Customer Id'] + network_features + ['is_fraud']].copy()
    df_network_only.to_csv('data/network_only.csv', index=False)
    print(f"  ✅ Network-only: {len(network_features)} features → data/network_only.csv")
    
    # 3. Combined
    df_merged.to_csv('data/combined_features.csv', index=False)
    print(f"  ✅ Combined: {len(all_features)} features → data/combined_features.csv")
    
    return {
        'transaction': transaction_features,
        'network': network_features,
        'combined': all_features
    }

def main():
    """Main execution"""
    # Paths - adjust based on current directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    transaction_path = os.path.join(current_dir, 'data', 'transaction_features.csv')
    network_path = os.path.join(current_dir, '..', 'data', 'network_features.csv')
    
    # Check if files exist
    if not os.path.exists(transaction_path):
        print(f"Error: Transaction features not found at {transaction_path}")
        print("Please run extract_transaction_features.py first")
        return
    
    if not os.path.exists(network_path):
        print(f"Error: Network features not found at {network_path}")
        print("Please ensure network_features.csv exists in data/ folder")
        return
    
    # Check for link prediction features
    lp_path = os.path.join(current_dir, '..', 'data', 'link_prediction_features.csv')
    has_lp_features = os.path.exists(lp_path)
    
    if has_lp_features:
        print(f"✓ Found link prediction features at {lp_path}")
    else:
        print(f"⚠️ Link prediction features not found (optional)")

    
    # Step 1: Load features
    df_transaction = load_transaction_features(transaction_path)
    df_network = load_network_features(network_path)
    
    # Step 2: Merge transaction + network
    df_merged = merge_features(df_transaction, df_network)
    
    # Step 2.5: Merge link prediction features if available
    if has_lp_features:
        print("\nMerging link prediction features...")
        df_lp = pd.read_csv(lp_path)
        
        # Rename customer_id to Customer Id
        if 'customer_id' in df_lp.columns:
            df_lp.rename(columns={'customer_id': 'Customer Id'}, inplace=True)
        
        # Select only LP features (exclude is_fraud)
        lp_cols = ['Customer Id', 'avg_jaccard_unmade', 'max_adamic_adar_unmade',
                   'num_high_sim_unmade', 'num_low_sim_made', 'anomaly_score']
        existing_lp_cols = [col for col in lp_cols if col in df_lp.columns]
        df_lp_selected = df_lp[existing_lp_cols].copy()
        
        # Merge
        df_merged = pd.merge(df_merged, df_lp_selected, on='Customer Id', how='left')
        print(f"  ✓ Added {len(existing_lp_cols)-1} link prediction features")
    
    # Step 3: Save 3 versions
    feature_dict = save_separate_datasets(df_merged)
    
    # Show sample
    print("\nSample of combined features:")
    print(df_merged.head())
    
    # Show fraud distribution
    fraud_customers = df_merged['is_fraud'].sum()
    total_customers = len(df_merged)
    print(f"\nFraud distribution in merged dataset:")
    print(f"  Fraud customers: {fraud_customers} ({fraud_customers/total_customers*100:.2f}%)")
    print(f"  Normal customers: {total_customers - fraud_customers}")
    
    print("\n✅ Feature merging complete!")
    print("\nNext step: Run train_compare.py to train and compare models")

if __name__ == '__main__':
    main()
