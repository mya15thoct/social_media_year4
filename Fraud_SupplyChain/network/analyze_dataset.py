
import pandas as pd
import warnings
warnings.filterwarnings('ignore')


def analyze_dataset():
    """Phân tích dataset và trả lời các câu hỏi"""
    
    print("="*80)
    print("PHÂN TÍCH DATACO SUPPLY CHAIN DATASET")
    print("="*80)
    
    # Đọc dữ liệu
    print("\n[1] Đọc dữ liệu...")
    df = pd.read_csv('data/DataCoSupplyChainDataset.csv', encoding='latin-1')
    print("✓ Đã đọc dữ liệu thành công!")
    
    # Câu hỏi 1: Số rows
    print("\n" + "-"*80)
    print("CÂU HỎI 1: Có bao nhiêu rows?")
    print("-"*80)
    num_rows = len(df)
    print(f"✓ Tổng số rows: {num_rows:,}")
    
    # Câu hỏi 2: Số columns
    print("\n" + "-"*80)
    print("CÂU HỎI 2: Có bao nhiêu columns?")
    print("-"*80)
    num_cols = len(df.columns)
    print(f"✓ Tổng số columns: {num_cols}")
    print(f"\nDanh sách các columns:")
    for i, col in enumerate(df.columns, 1):
        print(f"  {i:2d}. {col}")
    
    # Câu hỏi 3: Fraud label ở cột nào?
    print("\n" + "-"*80)
    print("CÂU HỎI 3: Fraud label ở cột nào?")
    print("-"*80)
    
    # Tìm các cột liên quan đến fraud/order status
    fraud_keywords = ['fraud', 'status', 'suspicious', 'late', 'delivery']
    potential_fraud_cols = []
    
    for col in df.columns:
        for keyword in fraud_keywords:
            if keyword.lower() in col.lower():
                if col not in potential_fraud_cols:
                    potential_fraud_cols.append(col)
    
    print(f"✓ Tìm thấy {len(potential_fraud_cols)} cột có thể liên quan đến fraud:")
    for col in potential_fraud_cols:
        unique_vals = df[col].nunique()
        print(f"\n  - Cột: '{col}'")
        print(f"    Số giá trị unique: {unique_vals}")
        print(f"    Các giá trị: {df[col].value_counts().to_dict()}")
    
    # Câu hỏi 4: Số unique customers
    print("\n" + "-"*80)
    print("CÂU HỎI 4: Có bao nhiêu unique customers?")
    print("-"*80)
    
    # Tìm cột customer
    customer_cols = [col for col in df.columns if 'customer' in col.lower() and 'id' in col.lower()]
    
    if customer_cols:
        customer_col = customer_cols[0]
        num_unique_customers = df[customer_col].nunique()
        print(f"✓ Cột customer: '{customer_col}'")
        print(f"✓ Số unique customers: {num_unique_customers:,}")
    else:
        print("❌ Không tìm thấy cột Customer ID")
    
    # Câu hỏi 5: Số unique products
    print("\n" + "-"*80)
    print("CÂU HỎI 5: Có bao nhiêu unique products?")
    print("-"*80)
    
    # Tìm cột product
    product_cols = [col for col in df.columns if 'product' in col.lower()]
    
    print(f"Tìm thấy {len(product_cols)} cột liên quan đến product:")
    for col in product_cols:
        num_unique = df[col].nunique()
        print(f"  - '{col}': {num_unique:,} unique values")
    
    # Câu hỏi 6: Fraud rate
    print("\n" + "-"*80)
    print("CÂU HỎI 6: Fraud rate là bao nhiêu %?")
    print("-"*80)
    
    # Kiểm tra các cột có thể dùng làm fraud indicator
    print("\nPhân tích fraud indicators:")
    
    # 1. Order Status
    if 'Order Status' in df.columns:
        print(f"\n1. Từ 'Order Status':")
        status_counts = df['Order Status'].value_counts()
        print(status_counts)
        
        suspicious_statuses = ['SUSPECTED_FRAUD', 'CANCELED', 'PAYMENT_REVIEW']
        fraud_count_status = df['Order Status'].isin(suspicious_statuses).sum()
        fraud_rate_status = (fraud_count_status / len(df)) * 100
        print(f"\n   Số orders với suspicious status: {fraud_count_status:,}")
        print(f"   Fraud rate (từ status): {fraud_rate_status:.2f}%")
    
    # 2. Late delivery risk
    if 'Late_delivery_risk' in df.columns:
        print(f"\n2. Từ 'Late_delivery_risk':")
        late_count = df['Late_delivery_risk'].sum()
        late_rate = (late_count / len(df)) * 100
        print(f"   Số orders có late delivery risk: {late_count:,}")
        print(f"   Late delivery rate: {late_rate:.2f}%")
    
    # 3. Benefit per order (negative = potential fraud)
    if 'Benefit per order' in df.columns:
        print(f"\n3. Từ 'Benefit per order' (negative benefit):")
        negative_benefit = (df['Benefit per order'] < 0).sum()
        negative_rate = (negative_benefit / len(df)) * 100
        print(f"   Số orders có benefit âm: {negative_benefit:,}")
        print(f"   Negative benefit rate: {negative_rate:.2f}%")
    
    # Tạo tổng hợp fraud label
    print("\n" + "-"*80)
    print("TẠO TỔNG HỢP FRAUD LABEL")
    print("-"*80)
    
    df['is_fraud'] = 0
    
    if 'Late_delivery_risk' in df.columns:
        df.loc[df['Late_delivery_risk'] == 1, 'is_fraud'] = 1
    
    if 'Order Status' in df.columns:
        suspicious_statuses = ['SUSPECTED_FRAUD', 'CANCELED', 'PAYMENT_REVIEW']
        df.loc[df['Order Status'].isin(suspicious_statuses), 'is_fraud'] = 1
    
    if 'Benefit per order' in df.columns:
        df.loc[df['Benefit per order'] < 0, 'is_fraud'] = 1
    
    total_fraud = df['is_fraud'].sum()
    fraud_rate = (total_fraud / len(df)) * 100
    
    print(f"\n✓ FRAUD RATE TỔNG HỢP:")
    print(f"  - Tổng số transactions: {len(df):,}")
    print(f"  - Số fraud transactions: {total_fraud:,}")
    print(f"  - Fraud rate: {fraud_rate:.2f}%")
    
    # Tóm tắt cuối cùng
    print("\n" + "="*80)
    print("TÓM TẮT KẾT QUẢ")
    print("="*80)
    print(f"1. Số rows: {num_rows:,}")
    print(f"2. Số columns: {num_cols}")
    print(f"3. Fraud label columns: {', '.join(potential_fraud_cols)}")
    if customer_cols:
        print(f"4. Số unique customers: {num_unique_customers:,}")
    print(f"5. Số unique products: {df[product_cols[0]].nunique():,} (từ cột '{product_cols[0]}')")
    print(f"6. Fraud rate: {fraud_rate:.2f}%")
    print("="*80)
    
    return df


if __name__ == "__main__":
    df = analyze_dataset()
