"""
BIPARTITE NETWORK VISUALIZATION FOR PAPER
==========================================
Generate Figure 1: Customer-Product Bipartite Network

Shows:
- Customers (circles) on left - color-coded by fraud status
- Products (squares) on right
- Edges connecting purchases
- Clear bipartite structure
"""

import pandas as pd
import networkx as nx
from networkx.algorithms import bipartite
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os

print("="*80)
print("BIPARTITE NETWORK VISUALIZATION")
print("="*80)

# Load data
script_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.join(script_dir, '..', '..')
data_path = os.path.join(root_dir, 'data', 'DataCoSupplyChainDataset.csv')

print(f"\n[1] Loading data from: {data_path}")
df = pd.read_csv(data_path, encoding='latin1')

print(f"Total orders: {len(df):,}")
print(f"Fraud orders: {df['Order Status'].eq('SUSPECTED_FRAUD').sum():,}")

# Sample data for visualization (full graph is too large)
# Take top products and sample of customers
print("\n[2] Sampling data for visualization...")

# Get top 30 most popular products
product_counts = df['Product Name'].value_counts().head(30)
top_products = product_counts.index.tolist()

# Filter to these products
df_sample = df[df['Product Name'].isin(top_products)].copy()

# Sample customers: take fraud customers + random normal customers
fraud_customers = df_sample[df_sample['Order Status'] == 'SUSPECTED_FRAUD']['Customer Id'].unique()
normal_customers = df_sample[df_sample['Order Status'] != 'SUSPECTED_FRAUD']['Customer Id'].unique()

# Take all fraud customers + sample of normal
n_normal_sample = min(200, len(normal_customers))
np.random.seed(42)
sampled_normal = np.random.choice(normal_customers, n_normal_sample, replace=False)

selected_customers = set(fraud_customers) | set(sampled_normal)
df_viz = df_sample[df_sample['Customer Id'].isin(selected_customers)].copy()

print(f"Sampled network:")
print(f"  Customers: {len(selected_customers):,} ({len(fraud_customers)} fraud, {len(sampled_normal)} normal)")
print(f"  Products: {len(top_products)}")
print(f"  Edges: {len(df_viz):,}")

# Create bipartite graph
print("\n[3] Building bipartite graph...")
G = nx.Graph()

# Add customer nodes (bipartite=0)
customer_fraud_status = {}
for cust_id in selected_customers:
    is_fraud = cust_id in fraud_customers
    G.add_node(f"C_{cust_id}", bipartite=0, node_type='customer', is_fraud=is_fraud)
    customer_fraud_status[f"C_{cust_id}"] = is_fraud

# Add product nodes (bipartite=1)
for prod in top_products:
    G.add_node(f"P_{prod[:20]}", bipartite=1, node_type='product')  # Truncate long names

# Add edges
for _, row in df_viz.iterrows():
    customer = f"C_{row['Customer Id']}"
    product = f"P_{row['Product Name'][:20]}"
    if customer in G.nodes and product in G.nodes:
        if G.has_edge(customer, product):
            G[customer][product]['weight'] += 1
        else:
            G.add_edge(customer, product, weight=1)

print(f"Graph created:")
print(f"  Nodes: {G.number_of_nodes()}")
print(f"  Edges: {G.number_of_edges()}")

# Separate nodes by type
customer_nodes = [n for n in G.nodes if G.nodes[n]['node_type'] == 'customer']
product_nodes = [n for n in G.nodes if G.nodes[n]['node_type'] == 'product']

fraud_customers_nodes = [n for n in customer_nodes if G.nodes[n]['is_fraud']]
normal_customers_nodes = [n for n in customer_nodes if not G.nodes[n]['is_fraud']]

print(f"  Customer nodes: {len(customer_nodes)} ({len(fraud_customers_nodes)} fraud)")
print(f"  Product nodes: {len(product_nodes)}")

# Create layout - bipartite layout
print("\n[4] Creating bipartite layout...")
pos = {}

# Customers on left (x=0)
customer_y = np.linspace(0, 1, len(customer_nodes))
for i, node in enumerate(customer_nodes):
    pos[node] = (0, customer_y[i])

# Products on right (x=1)
product_y = np.linspace(0, 1, len(product_nodes))
for i, node in enumerate(product_nodes):
    pos[node] = (1, product_y[i])

# Create visualization
print("\n[5] Drawing network...")
fig, ax = plt.subplots(figsize=(16, 12))

# Draw edges (light gray, thin)
nx.draw_networkx_edges(
    G, pos,
    alpha=0.15,
    width=0.3,
    edge_color='gray',
    ax=ax
)

# Draw normal customers (blue circles)
nx.draw_networkx_nodes(
    G, pos,
    nodelist=normal_customers_nodes,
    node_color='#3498db',  # Blue
    node_shape='o',
    node_size=50,
    alpha=0.7,
    ax=ax,
    label='Normal Customer'
)

# Draw fraud customers (red circles)
nx.draw_networkx_nodes(
    G, pos,
    nodelist=fraud_customers_nodes,
    node_color='#e74c3c',  # Red
    node_shape='o',
    node_size=80,
    alpha=0.9,
    ax=ax,
    label='Fraud Customer'
)

# Draw products (green squares)
nx.draw_networkx_nodes(
    G, pos,
    nodelist=product_nodes,
    node_color='#2ecc71',  # Green
    node_shape='s',
    node_size=100,
    alpha=0.8,
    ax=ax,
    label='Product'
)

# Add labels
ax.text(-0.05, 1.02, 'CUSTOMERS', fontsize=14, fontweight='bold', ha='center')
ax.text(1.05, 1.02, 'PRODUCTS', fontsize=14, fontweight='bold', ha='center')

# Title and legend
ax.set_title('Bipartite Network: Customer-Product Relationships\n(Fraud Detection in Supply Chain)', 
             fontsize=16, fontweight='bold', pad=20)

ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.02), ncol=3, fontsize=11)

# Remove axes
ax.axis('off')

# Add network statistics as text box
stats_text = f"""Network Statistics:
• Customers: {len(customer_nodes):,} ({len(fraud_customers_nodes)} fraudulent)
• Products: {len(product_nodes)}
• Relationships: {G.number_of_edges():,}
• Fraud Rate: {len(fraud_customers_nodes)/len(customer_nodes)*100:.1f}%"""

props = dict(boxstyle='round', facecolor='wheat', alpha=0.3)
ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
        verticalalignment='top', bbox=props, family='monospace')

plt.tight_layout()

# Save figure
output_dir = os.path.join(script_dir, 'figures')
os.makedirs(output_dir, exist_ok=True)

png_path = os.path.join(output_dir, 'figure1_bipartite_network.png')
pdf_path = os.path.join(output_dir, 'figure1_bipartite_network.pdf')

plt.savefig(png_path, dpi=300, bbox_inches='tight', facecolor='white')
plt.savefig(pdf_path, bbox_inches='tight', facecolor='white')

print(f"\n[6] Saved visualizations:")
print(f"  PNG: {png_path}")
print(f"  PDF: {pdf_path}")

print("\n" + "="*80)
print("✓ VISUALIZATION COMPLETE!")
print("="*80)
print("\nFor paper: Use the PDF version for best quality")

plt.show()
