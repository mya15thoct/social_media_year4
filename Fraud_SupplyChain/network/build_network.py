"""
BƯỚC 2.2: XÂY DỰNG BIPARTITE NETWORK
Từ edge list → NetworkX graph object
"""
import pandas as pd
import networkx as nx
import pickle
import warnings
warnings.filterwarnings('ignore')


def build_bipartite_network():
    """Xây dựng bipartite network từ edge list"""
    
    print("="*80)
    print("BƯỚC 2.2: XÂY DỰNG BIPARTITE NETWORK")
    print("="*80)
    
    # Đọc edge list
    print("\n[1] Đọc edge list...")
    df = pd.read_csv('data/edgelist.csv')
    print(f"✓ Đã đọc {len(df):,} edges")
    
    # Tạo bipartite graph
    print("\n[2] Tạo bipartite graph...")
    G = nx.Graph()
    
    # Thêm customer nodes (set 0)
    customer_nodes = df['customer_id'].unique()
    print(f"   Thêm {len(customer_nodes):,} customer nodes...")
    for cust in customer_nodes:
        G.add_node(f'C_{cust}', bipartite=0)
    
    # Thêm product nodes (set 1)
    product_nodes = df['product_id'].unique()
    print(f"   Thêm {len(product_nodes):,} product nodes...")
    for prod in product_nodes:
        G.add_node(f'P_{prod}', bipartite=1)
    
    # Thêm edges với attributes
    print(f"\n[3] Thêm edges với attributes...")
    edge_count = 0
    
    # Group by customer-product pairs để aggregate
    grouped = df.groupby(['customer_id', 'product_id']).agg({
        'sales': 'sum',
        'quantity': 'sum',
        'is_fraud': 'max'  # Nếu có 1 transaction fraud thì edge = fraud
    }).reset_index()
    
    for _, row in grouped.iterrows():
        customer_node = f"C_{row['customer_id']}"
        product_node = f"P_{row['product_id']}"
        
        G.add_edge(
            customer_node,
            product_node,
            weight=1,  # Số lần mua (có thể adjust)
            total_sales=float(row['sales']),
            total_quantity=int(row['quantity'])
        )
        edge_count += 1
    
    print(f"✓ Đã thêm {edge_count:,} unique edges")
    
    # Thêm node attributes: fraud count và normal count
    print("\n[4] Tính fraud count cho mỗi node...")
    
    # Customer nodes
    for cust_id in customer_nodes:
        cust_node = f'C_{cust_id}'
        cust_data = df[df['customer_id'] == cust_id]
        fraud_count = cust_data['is_fraud'].sum()
        normal_count = len(cust_data) - fraud_count
        G.nodes[cust_node]['fraud_count'] = int(fraud_count)
        G.nodes[cust_node]['normal_count'] = int(normal_count)
    
    # Product nodes
    for prod_id in product_nodes:
        prod_node = f'P_{prod_id}'
        prod_data = df[df['product_id'] == prod_id]
        fraud_count = prod_data['is_fraud'].sum()
        normal_count = len(prod_data) - fraud_count
        G.nodes[prod_node]['fraud_count'] = int(fraud_count)
        G.nodes[prod_node]['normal_count'] = int(normal_count)
    
    # Network statistics
    print("\n[5] Thống kê network...")
    print(f"   Total nodes: {G.number_of_nodes():,}")
    print(f"   Total edges: {G.number_of_edges():,}")
    print(f"   Density: {nx.density(G):.6f}")
    print(f"   Is bipartite: {nx.bipartite.is_bipartite(G)}")
    print(f"   Is connected: {nx.is_connected(G)}")
    
    if not nx.is_connected(G):
        components = list(nx.connected_components(G))
        print(f"   Number of components: {len(components)}")
        largest = max(components, key=len)
        print(f"   Largest component size: {len(largest):,} nodes")
    
    # Lưu graph object
    print("\n[6] Lưu graph object...")
    with open('data/bipartite_graph.gpickle', 'wb') as f:
        pickle.dump(G, f)
    print("✓ Đã lưu: data/bipartite_graph.gpickle")
    
    # Lưu graph info (metadata nhỏ gọn)
    graph_info = {
        'num_nodes': G.number_of_nodes(),
        'num_customers': len(customer_nodes),
        'num_products': len(product_nodes),
        'num_edges': G.number_of_edges(),
        'num_fraud_customers': int(df.groupby('customer_id')['is_fraud'].max().sum()),
        'density': nx.density(G),
        'is_bipartite': nx.bipartite.is_bipartite(G),
        'is_connected': nx.is_connected(G),
        'avg_degree': sum(dict(G.degree()).values()) / G.number_of_nodes()
    }
    
    with open('data/graph_info.pkl', 'wb') as f:
        pickle.dump(graph_info, f)
    print("✓ Đã lưu: data/graph_info.pkl")
    
    print("\n" + "="*80)
    print("HOÀN TẤT XÂY DỰNG NETWORK!")
    print("="*80)
    print("\nNetwork đã sẵn sàng để:")
    print("  → Tính network features (degree, betweenness, closeness)")
    print("  → Phát hiện communities")
    print("  → Visualize trong Gephi")
    
    return G, graph_info


if __name__ == "__main__":
    G, info = build_bipartite_network()
