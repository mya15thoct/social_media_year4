"""
BƯỚC 2.1: CHUẨN BỊ EDGE LIST
Tạo file edgelist đơn giản để build network
"""
import pandas as pd
import warnings
warnings.filterwarnings('ignore')


def create_edgelist():
    """Tạo edge list từ dataset gốc"""
    
    print("="*80)
    print("BƯỚC 2.1: TẠO EDGE LIST CHO NETWORK")
    print("="*80)
    
    # Đọc dữ liệu gốc
    print("\n[1] Đọc dữ liệu gốc...")
    df = pd.read_csv('data/DataCoSupplyChainDataset.csv', encoding='latin-1')
    print(f"✓ Đã đọc {len(df):,} rows và {len(df.columns)} columns")
    
    # Chọn các cột cần thiết
    print("\n[2] Chọn các cột cần thiết cho edge list...")
    
    columns_needed = [
        'Customer Id',           # Customer node
        'Product Card Id',       # Product node
        'Sales',                 # Trọng số của edge
        'Order Item Quantity',   # Số lượng
        'order date (DateOrders)',  # Thời gian
        'Order Status'           # Để xác định fraud
    ]
    
    # Kiểm tra xem các cột có tồn tại không
    missing_cols = [col for col in columns_needed if col not in df.columns]
    if missing_cols:
        print(f"❌ Thiếu các cột: {missing_cols}")
        return None
    
    # Tạo edge list
    edgelist = df[columns_needed].copy()
    
    print(f"✓ Đã chọn {len(columns_needed)} cột:")
    for col in columns_needed:
        print(f"  - {col}")
    
    # Tạo fraud label
    print("\n[3] Tạo fraud label...")
    print("  Sử dụng: Order Status = 'SUSPECTED_FRAUD'")
    
    edgelist['is_fraud'] = 0
    edgelist.loc[edgelist['Order Status'] == 'SUSPECTED_FRAUD', 'is_fraud'] = 1
    
    fraud_count = edgelist['is_fraud'].sum()
    fraud_rate = (fraud_count / len(edgelist)) * 100
    
    print(f"  ✓ Fraud cases: {fraud_count:,} ({fraud_rate:.2f}%)")
    print(f"  ✓ Normal cases: {len(edgelist) - fraud_count:,} ({100-fraud_rate:.2f}%)")
    
    # Đổi tên cột cho dễ hiểu
    print("\n[4] Đổi tên cột...")
    edgelist = edgelist.rename(columns={
        'Customer Id': 'customer_id',
        'Product Card Id': 'product_id',
        'Sales': 'sales',
        'Order Item Quantity': 'quantity',
        'order date (DateOrders)': 'order_date',
        'Order Status': 'order_status'
    })
    
    # Thông tin về edge list
    print("\n[5] Thông tin về edge list:")
    print(f"  - Tổng số edges (transactions): {len(edgelist):,}")
    print(f"  - Unique customers: {edgelist['customer_id'].nunique():,}")
    print(f"  - Unique products: {edgelist['product_id'].nunique():,}")
    print(f"  - Tổng sales: ${edgelist['sales'].sum():,.2f}")
    print(f"  - Tổng quantity: {edgelist['quantity'].sum():,.0f}")
    
    # Hiển thị mẫu
    print("\n[6] Mẫu dữ liệu (5 dòng đầu):")
    print(edgelist.head())
    
    print("\n[7] Thống kê cơ bản:")
    print(edgelist.describe())
    
    # Lưu file
    print("\n[8] Lưu edge list...")
    output_path = 'data/edgelist.csv'
    edgelist.to_csv(output_path, index=False)
    print(f"✓ Đã lưu vào: {output_path}")
    
    # Tóm tắt
    print("\n" + "="*80)
    print("TÓM TẮT EDGE LIST")
    print("="*80)
    print(f"File: edgelist.csv")
    print(f"Số cột: {len(edgelist.columns)}")
    print(f"Số dòng: {len(edgelist):,}")
    print(f"\nCác cột:")
    for i, col in enumerate(edgelist.columns, 1):
        print(f"  {i}. {col}")
    
    print(f"\nFraud distribution:")
    print(f"  - Normal: {(edgelist['is_fraud']==0).sum():,} ({(edgelist['is_fraud']==0).sum()/len(edgelist)*100:.2f}%)")
    print(f"  - Fraud: {(edgelist['is_fraud']==1).sum():,} ({(edgelist['is_fraud']==1).sum()/len(edgelist)*100:.2f}%)")
    
    print("\n✓ HOÀN THÀNH BƯỚC 2.1!")
    print("="*80)
    
    return edgelist


if __name__ == "__main__":
    edgelist = create_edgelist()
