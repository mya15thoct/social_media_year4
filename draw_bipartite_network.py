"""
Draw Bipartite Network from DataCo Supply Chain Dataset
Visualizes Customer-Product relationships with fraud highlighting
"""

import pandas as pd
import networkx as nx
from networkx.algorithms import bipartite
import matplotlib.pyplot as plt
import numpy as np

def load_data():
    """Load DataCo dataset"""
    print("Loading DataCo Supply Chain Dataset...")
    df = pd.read_csv('data/DataCoSupplyChainDataset.csv', encoding='latin-1')
    print(f"Total transactions: {len(df):,}")
    print(f"Fraud rate: {df['Order Status'].value_counts(normalize=True).get('SUSPECTED_FRAUD', 0)*100:.2f}%")
    return df

def create_bipartite_graph(df, sample_size=500):
    """Create bipartite graph from customer-product relationships"""
    print(f"\nCreating bipartite graph (sample size: {sample_size})...")
    
    # Sample data for visualization (full graph would be too dense)
    df_sample = df.sample(n=min(sample_size, len(df)), random_state=42)
    
    # Create bipartite graph
    B = nx.Graph()
    
    # Add nodes
    customers = df_sample['Customer Id'].unique()
    products = df_sample['Product Name'].unique()
    
    # Limit products to most common ones for cleaner visualization
    product_counts = df_sample['Product Name'].value_counts()
    top_products = product_counts.head(4).index.tolist()
    
    # Filter to only include transactions with top products
    df_filtered = df_sample[df_sample['Product Name'].isin(top_products)]
    
    customers = df_filtered['Customer Id'].unique()
    products = top_products
    
    print(f"Customers: {len(customers)}")
    print(f"Products: {len(products)}")
    
    # Add customer nodes (bipartite=0)
    B.add_nodes_from([f"C_{c}" for c in customers], bipartite=0)
    
    # Add product nodes (bipartite=1)
    B.add_nodes_from([f"P_{p}" for p in products], bipartite=1)
    
    # Add edges and track fraud
    fraud_edges = []
    normal_edges = []
    
    for _, row in df_filtered.iterrows():
        customer = f"C_{row['Customer Id']}"
        product = f"P_{row['Product Name']}"
        
        if row['Order Status'] == 'SUSPECTED_FRAUD':
            fraud_edges.append((customer, product))
        else:
            normal_edges.append((customer, product))
    
    B.add_edges_from(normal_edges)
    B.add_edges_from(fraud_edges)
    
    print(f"Total edges: {len(normal_edges) + len(fraud_edges)}")
    print(f"Fraud edges: {len(fraud_edges)}")
    print(f"Normal edges: {len(normal_edges)}")
    
    # Get actual counts from full dataset
    total_customers = df['Customer Id'].nunique()
    total_products = df['Product Name'].nunique()
    total_edges = len(df)
    
    return B, fraud_edges, normal_edges, (total_customers, total_products, total_edges)

def draw_bipartite_network(B, fraud_edges, normal_edges, full_stats):
    """Draw bipartite network visualization"""
    print("\nDrawing bipartite network...")
    
    fig, ax = plt.subplots(figsize=(16, 10))
    
    # Separate customer and product nodes
    customer_nodes = [n for n, d in B.nodes(data=True) if d['bipartite'] == 0]
    product_nodes = [n for n, d in B.nodes(data=True) if d['bipartite'] == 1]
    
    # Position nodes
    pos = {}
    
    # Customer nodes on the left
    y_spacing_customers = 8 / max(len(customer_nodes), 1)
    for i, node in enumerate(customer_nodes):
        pos[node] = (1, 9 - i * y_spacing_customers)
    
    # Product nodes on the right
    y_spacing_products = 8 / max(len(product_nodes), 1)
    for i, node in enumerate(product_nodes):
        pos[node] = (11, 9 - i * y_spacing_products)
    
    # Draw edges (normal first, fraud on top)
    nx.draw_networkx_edges(B, pos, edgelist=normal_edges, 
                          edge_color='gray', alpha=0.3, width=1,
                          ax=ax)
    
    nx.draw_networkx_edges(B, pos, edgelist=fraud_edges,
                          edge_color='red', alpha=0.6, width=2,
                          ax=ax, style='solid')
    
    # Draw customer nodes
    # Identify fraud customers
    fraud_customers = set([edge[0] for edge in fraud_edges])
    normal_customers = [n for n in customer_nodes if n not in fraud_customers]
    
    nx.draw_networkx_nodes(B, pos, nodelist=normal_customers,
                          node_color='lightblue', node_size=800,
                          node_shape='o', edgecolors='black', linewidths=2,
                          ax=ax)
    
    if fraud_customers:
        nx.draw_networkx_nodes(B, pos, nodelist=list(fraud_customers),
                              node_color='#ffcccc', node_size=800,
                              node_shape='o', edgecolors='red', linewidths=3,
                              ax=ax)
    
    # Draw product nodes
    nx.draw_networkx_nodes(B, pos, nodelist=product_nodes,
                          node_color='lightgreen', node_size=1000,
                          node_shape='o', edgecolors='black', linewidths=2,
                          ax=ax)
    
    # Draw labels
    customer_labels = {node: node.replace('C_', 'C') for node in customer_nodes[:5]}  # Show first 5
    product_labels = {node: node.replace('P_', '').split()[0][:8] for node in product_nodes}
    
    nx.draw_networkx_labels(B, pos, labels=customer_labels,
                           font_size=9, font_weight='bold',
                           ax=ax)
    
    nx.draw_networkx_labels(B, pos, labels=product_labels,
                           font_size=10, font_weight='bold',
                           ax=ax)
    
    # Add header boxes
    # Customer header
    ax.add_patch(plt.Rectangle((0.2, 10), 1.6, 0.8, 
                               facecolor='lightblue', edgecolor='black',
                               linewidth=2, alpha=0.7))
    ax.text(1.0, 10.4, f'CUSTOMER NODES\n({full_stats[0]:,} nodes)', 
            ha='center', va='center', fontsize=11, fontweight='bold')
    
    # Product header
    ax.add_patch(plt.Rectangle((10.2, 10), 1.6, 0.8,
                               facecolor='lightgreen', edgecolor='black',
                               linewidth=2, alpha=0.7))
    ax.text(11.0, 10.4, f'PRODUCT NODES\n({full_stats[1]} nodes)',
            ha='center', va='center', fontsize=11, fontweight='bold')
    
    # Add edge info in center
    ax.text(6, 5.5, f'EDGES (Transactions)\n{full_stats[2]:,} unique customer-product pairs',
            ha='center', va='center', fontsize=11, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white', 
                     edgecolor='black', linewidth=2))
    
    # Add legend
    legend_elements = [
        plt.Line2D([0], [0], color='gray', linewidth=2, alpha=0.5, label='Normal Transaction'),
        plt.Line2D([0], [0], color='red', linewidth=3, label='Fraud Transaction'),
        plt.scatter([], [], s=200, c='lightblue', edgecolors='black', linewidths=2, label='Normal Customer'),
        plt.scatter([], [], s=200, c='#ffcccc', edgecolors='red', linewidths=3, label='Fraud Customer'),
        plt.scatter([], [], s=200, c='lightgreen', edgecolors='black', linewidths=2, label='Product'),
    ]
    ax.legend(handles=legend_elements, loc='lower center', 
             ncol=5, fontsize=10, frameon=True, fancybox=True)
    
    # Title
    ax.set_title('Bipartite Network Structure: Customer-Product Relationships',
                fontsize=16, fontweight='bold', pad=20)
    
    ax.set_xlim(-0.5, 12.5)
    ax.set_ylim(0, 11)
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('bipartite_network_dataco.png', dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    print("✓ Saved: bipartite_network_dataco.png")
    plt.show()

def main():
    """Main function"""
    # Load data
    df = load_data()
    
    # Create bipartite graph
    B, fraud_edges, normal_edges, full_stats = create_bipartite_graph(df, sample_size=500)
    
    # Draw network
    draw_bipartite_network(B, fraud_edges, normal_edges, full_stats)
    
    print("\n" + "="*70)
    print("Bipartite network visualization completed!")
    print("="*70)

if __name__ == "__main__":
    main()
