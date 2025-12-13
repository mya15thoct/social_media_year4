import pandas as pd
paths = [
    r'c:\IT\student\fraud_supplychain\Fraud_SupplyChain\data\combined_features.csv',
    r'c:\IT\student\fraud_supplychain\Fraud_SupplyChain\data\transaction_only.csv',
    r'c:\IT\student\fraud_supplychain\Fraud_SupplyChain\data\network_only.csv'
]
for p in paths:
    try:
        df = pd.read_csv(p, encoding='latin-1')
    except Exception as e:
        try:
            df = pd.read_csv(p)
        except Exception as e2:
            print(f'ERROR reading {p}:', e2)
            continue
    print(f"File: {p}")
    print('  shape:', df.shape)
    if 'Order Status' in df.columns:
        print('  Order Status counts:')
        print(df['Order Status'].value_counts(dropna=False))
    if 'Customer Id' in df.columns and 'Product Name' in df.columns:
        print('  Unique customers:', df['Customer Id'].nunique())
        print('  Unique products:', df['Product Name'].nunique())
    print('-'*60)

# Also check train/test split if possible
try:
    from Fraud_SupplyChain.combined_model import main_ensemble as me
    print('Found main_ensemble module')
except Exception:
    pass
