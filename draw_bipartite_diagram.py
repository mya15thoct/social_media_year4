"""
Bipartite Network Visualization: Customer-Product Relationships
Using DataCo Supply Chain Dataset - Simple Diagram Style
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Circle, FancyArrowPatch
import numpy as np

def draw_bipartite_diagram():
    """
    Draw bipartite network diagram showing customer-product relationships from DataCo dataset
    """
    # Load data
    print("Loading DataCo dataset...")
    df = pd.read_csv('data/DataCoSupplyChainDataset.csv', encoding='ISO-8859-1')
    
    # Get network statistics
    n_customers = df['Customer Id'].nunique()
    n_products = df['Product Name'].nunique()
    n_edges = len(df)
    n_unique_pairs = df[['Customer Id', 'Product Name']].drop_duplicates().shape[0]
    
    # Identify fraud customers
    fraud_customers = df[df['Order Status'] == 'SUSPECTED_FRAUD']['Customer Id'].unique()
    n_fraud_customers = len(fraud_customers)
    
    print(f"Network Statistics:")
    print(f"  Customers: {n_customers:,}")
    print(f"  Products: {n_products:,}")
    print(f"  Total Transactions: {n_edges:,}")
    print(f"  Unique Customer-Product Pairs: {n_unique_pairs:,}")
    print(f"  Fraud Customers: {n_fraud_customers:,}")
    
    # Create figure
    fig, ax = plt.subplots(figsize=(16, 10))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Title
    ax.text(8, 9.5, 'Bipartite Network Structure: Customer–Product Relationships', 
            fontsize=18, fontweight='bold', ha='center')
    
    # ========== LEFT SIDE: CUSTOMER NODES ==========
    # Header box for customers
    customer_header = FancyBboxPatch(
        (0.5, 8.5), 3.5, 0.8,
        boxstyle="round,pad=0.1",
        edgecolor='#2c3e50',
        facecolor='#3498db',
        linewidth=2,
        alpha=0.3
    )
    ax.add_patch(customer_header)
    ax.text(2.25, 8.9, 'CUSTOMER NODES', fontsize=12, fontweight='bold', ha='center')
    ax.text(2.25, 8.65, f'({n_customers:,} nodes)', fontsize=10, ha='center')
    
    # Sample customer nodes (show 3 normal + 1 fraud)
    customer_positions = [
        {'y': 7, 'label': 'C1', 'fraud': False},
        {'y': 5.5, 'label': 'C2', 'fraud': False},
        {'y': 4, 'label': 'C3', 'fraud': False},
        {'y': 2, 'label': 'C_fraud', 'fraud': True},
    ]
    
    customer_x = 2.25
    for cust in customer_positions:
        color = '#ffcccb' if cust['fraud'] else '#aed6f1'
        edge_color = '#e74c3c' if cust['fraud'] else '#3498db'
        
        circle = Circle(
            (customer_x, cust['y']), 0.4,
            edgecolor=edge_color,
            facecolor=color,
            linewidth=2.5 if cust['fraud'] else 2,
            zorder=10
        )
        ax.add_patch(circle)
        ax.text(customer_x, cust['y'], cust['label'], 
                fontsize=11, fontweight='bold', ha='center', va='center', zorder=11)
    
    # ========== RIGHT SIDE: PRODUCT NODES ==========
    # Header box for products
    product_header = FancyBboxPatch(
        (12, 8.5), 3.5, 0.8,
        boxstyle="round,pad=0.1",
        edgecolor='#27ae60',
        facecolor='#a9dfbf',
        linewidth=2,
        alpha=0.3
    )
    ax.add_patch(product_header)
    ax.text(13.75, 8.9, 'PRODUCT NODES', fontsize=12, fontweight='bold', ha='center')
    ax.text(13.75, 8.65, f'({n_products} nodes)', fontsize=10, ha='center')
    
    # Sample product nodes
    product_positions = [
        {'y': 7, 'label': 'P1'},
        {'y': 5.5, 'label': 'P2'},
        {'y': 4, 'label': 'P3'},
        {'y': 2, 'label': 'P4'},
    ]
    
    product_x = 13.75
    for prod in product_positions:
        circle = Circle(
            (product_x, prod['y']), 0.4,
            edgecolor='#27ae60',
            facecolor='#d5f4e6',
            linewidth=2,
            zorder=10
        )
        ax.add_patch(circle)
        ax.text(product_x, prod['y'], prod['label'], 
                fontsize=11, fontweight='bold', ha='center', va='center', zorder=11)
    
    # ========== EDGES (TRANSACTIONS) ==========
    # Draw edges connecting customers to products
    edges = [
        # Normal transactions (gray)
        {'from': (customer_x, 7), 'to': (product_x, 7), 'fraud': False},
        {'from': (customer_x, 5.5), 'to': (product_x, 5.5), 'fraud': False},
        {'from': (customer_x, 4), 'to': (product_x, 4), 'fraud': False},
        # Fraud transaction (red)
        {'from': (customer_x, 2), 'to': (product_x, 2), 'fraud': True},
    ]
    
    for edge in edges:
        color = '#e74c3c' if edge['fraud'] else '#95a5a6'
        width = 2.5 if edge['fraud'] else 1.5
        alpha = 0.8 if edge['fraud'] else 0.5
        
        arrow = FancyArrowPatch(
            edge['from'], edge['to'],
            arrowstyle='-',
            color=color,
            linewidth=width,
            alpha=alpha,
            zorder=5
        )
        ax.add_patch(arrow)
    
    # Add arrow pointing to fraud transaction
    if True:  # Fraud edge exists
        # Small arrow pointing to P4
        arrow_start = (product_x + 0.5, 2.3)
        arrow_end = (product_x + 0.1, 2.1)
        arrow = FancyArrowPatch(
            arrow_start, arrow_end,
            arrowstyle='->',
            color='#e74c3c',
            linewidth=2,
            mutation_scale=20,
            zorder=15
        )
        ax.add_patch(arrow)
        
        # Another arrow
        arrow_start2 = (product_x + 0.5, 3.7)
        arrow_end2 = (product_x + 0.1, 3.9)
        arrow2 = FancyArrowPatch(
            arrow_start2, arrow_end2,
            arrowstyle='->',
            color='#e74c3c',
            linewidth=2,
            mutation_scale=20,
            zorder=15
        )
        ax.add_patch(arrow2)
    
    # ========== CENTER INFO BOX ==========
    info_box = FancyBboxPatch(
        (5.5, 4.5), 5, 1.5,
        boxstyle="round,pad=0.15",
        edgecolor='#34495e',
        facecolor='white',
        linewidth=2,
        zorder=20,
        alpha=0.95
    )
    ax.add_patch(info_box)
    
    ax.text(8, 5.6, 'EDGES (Transactions)', 
            fontsize=13, fontweight='bold', ha='center', zorder=21)
    ax.text(8, 5.2, f'{n_unique_pairs:,} unique customer–product pairs', 
            fontsize=11, ha='center', zorder=21, style='italic')
    ax.text(8, 4.85, f'Total: {n_edges:,} transactions', 
            fontsize=10, ha='center', zorder=21, color='#7f8c8d')
    
    # ========== LEGEND ==========
    legend_elements = [
        mpatches.Circle((0, 0), 0.3, edgecolor='#3498db', facecolor='#aed6f1', 
                       linewidth=2, label='Normal Customer'),
        mpatches.Circle((0, 0), 0.3, edgecolor='#e74c3c', facecolor='#ffcccb', 
                       linewidth=2.5, label='Fraud Customer'),
        mpatches.Circle((0, 0), 0.3, edgecolor='#27ae60', facecolor='#d5f4e6', 
                       linewidth=2, label='Product'),
        mpatches.Patch(color='#95a5a6', alpha=0.5, label='Normal Transaction'),
        mpatches.Patch(color='#e74c3c', alpha=0.8, label='Fraud Transaction'),
    ]
    
    ax.legend(handles=legend_elements, loc='lower center', 
             bbox_to_anchor=(0.5, -0.05), ncol=5, frameon=True, 
             fontsize=10, edgecolor='black')
    
    # ========== STATISTICS ANNOTATIONS ==========
    # Left annotation
    ax.text(0.5, 1, f'Fraud Rate: {n_fraud_customers/n_customers*100:.1f}%', 
            fontsize=9, style='italic', color='#e74c3c')
    
    # Right annotation  
    ax.text(12, 1, f'Avg products per customer: {n_unique_pairs/n_customers:.1f}', 
            fontsize=9, style='italic', color='#27ae60')
    
    plt.tight_layout()
    plt.savefig('bipartite_network_dataco.png', dpi=300, bbox_inches='tight')
    print("\nFigure saved as: bipartite_network_dataco.png")
    plt.show()

if __name__ == "__main__":
    draw_bipartite_diagram()
