"""
Clean corrupted data in combined_features.csv
"""
import pandas as pd
import numpy as np

print("Cleaning combined_features.csv...")

# Load with no type inference
df = pd.read_csv('data/combined_features.csv', dtype=str)
print(f"Loaded: {df.shape}")

# Identify numeric columns (all except Customer Id and is_fraud)
numeric_cols = [c for c in df.columns if c not in ['Customer Id', 'is_fraud']]

print(f"\nCleaning {len(numeric_cols)} numeric columns...")

# Clean each column
for col in numeric_cols:
    # Replace corrupted values (containing '/')
    df[col] = df[col].apply(lambda x: np.nan if isinstance(x, str) and '/' in x else x)
    # Convert to float
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Fill NaN with 0
df[numeric_cols] = df[numeric_cols].fillna(0)

# Convert is_fraud to int
df['is_fraud'] = pd.to_numeric(df['is_fraud'], errors='coerce').fillna(0).astype(int)

# Save cleaned version
df.to_csv('data/combined_features.csv', index=False)

print(f"\n✓ Cleaned and saved!")
print(f"Final shape: {df.shape}")
print(f"Data types: {df.dtypes.value_counts()}")
print(f"\nSample:")
print(df.head())
