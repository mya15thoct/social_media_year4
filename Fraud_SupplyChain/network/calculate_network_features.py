"""
Calculate Network Features

Extract centrality measures and community detection from bipartite graph.
"""
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import pickle
import networkx as nx
import pandas as pd
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')


def calculate_network_features():
    """Calculate network features for each customer"""
    
    print("="*80)
    print("CALCULATING NETWORK FEATURES")
    print("="*80)
    
    # Load network
    print("\n[1] Loading bipartite network...")
    with open('data/bipartite_graph.gpickle', 'rb') as f:
        G = pickle.load(f)
    
    print(f"Network loaded successfully:")
    print(f"  - Nodes: {G.number_of_nodes():,}")
    print(f"  - Edges: {G.number_of_edges():,}")
    
    # Filter customer nodes
    customer_nodes = [n for n in G.nodes() if n.startswith('C_')]
    print(f"  - Customer nodes: {len(customer_nodes):,}")
    
    # 1. DEGREE CENTRALITY
    print("\n[2] Calculating Degree Centrality...")
    print("  (Measures number of connections for each node)")
    
    degree_centrality = nx.degree_centrality(G)
    
    # Extract only customers
    degree_dict = {node: degree_centrality[node] for node in customer_nodes}
    
    print(f"  Calculated degree centrality for {len(degree_dict):,} customers")
    print(f"  - Min: {min(degree_dict.values()):.6f}")
    print(f"  - Max: {max(degree_dict.values()):.6f}")
    print(f"  - Mean: {sum(degree_dict.values())/len(degree_dict):.6f}")
    
    # 2. BETWEENNESS CENTRALITY
    print("\n[3] Calculating Betweenness Centrality...")
    print("  (Measures role as bridge between nodes)")
    print("  This may take a few minutes...")
    
    # Use sampling to speed up
    k = min(5000, G.number_of_nodes())
    betweenness_centrality = nx.betweenness_centrality(G, k=k)
    
    # Extract only customers
    betweenness_dict = {node: betweenness_centrality[node] for node in customer_nodes}
    
    print(f"  Calculated betweenness centrality for {len(betweenness_dict):,} customers")
    print(f"  - Min: {min(betweenness_dict.values()):.6f}")
    print(f"  - Max: {max(betweenness_dict.values()):.6f}")
    print(f"  - Mean: {sum(betweenness_dict.values())/len(betweenness_dict):.6f}")
    
    # 3. CLOSENESS CENTRALITY
    print("\n[4] Calculating Closeness Centrality...")
    print("  (Measures average distance to other nodes)")
    
    # Network is not connected, so calculate for each component
    # Or use closeness for disconnected graph
    closeness_dict = {}
    
    print("  Calculating closeness for each customer...")
    for node in tqdm(customer_nodes, desc="  Progress"):
        try:
            # Calculate closeness only within node's component
            closeness_dict[node] = nx.closeness_centrality(G, node)
        except:
            closeness_dict[node] = 0.0
    
    print(f"  Calculated closeness centrality for {len(closeness_dict):,} customers")
    print(f"  - Min: {min(closeness_dict.values()):.6f}")
    print(f"  - Max: {max(closeness_dict.values()):.6f}")
    print(f"  - Mean: {sum(closeness_dict.values())/len(closeness_dict):.6f}")
    
    # 4. PAGERANK
    print("\n[5] Calculating PageRank...")
    print("  (Random walk with teleportation - Lecture 4)")
    print("  Calculating PageRank...")
    
    pagerank_dict_full = nx.pagerank(G, alpha=0.85, max_iter=100)
    
    # Extract only customers
    pagerank_dict = {node: pagerank_dict_full[node] for node in customer_nodes}
    
    print(f"  Calculated PageRank for {len(pagerank_dict):,} customers")
    print(f"  - Min: {min(pagerank_dict.values()):.8f}")
    print(f"  - Max: {max(pagerank_dict.values()):.8f}")
    print(f"  - Mean: {sum(pagerank_dict.values())/len(pagerank_dict):.8f}")
    
    # 5. EIGENVECTOR CENTRALITY
    print("\n[6] Calculating Eigenvector Centrality...")
    print("  (Node is important if connected to important nodes - Lecture 4)")
    print("  Calculating Eigenvector Centrality...")
    
    try:
        eigenvector_dict_full = nx.eigenvector_centrality(G, max_iter=200, tol=1e-06)
        eigenvector_dict = {node: eigenvector_dict_full[node] for node in customer_nodes}
        
        print(f"  Calculated Eigenvector Centrality for {len(eigenvector_dict):,} customers")
        print(f"  - Min: {min(eigenvector_dict.values()):.8f}")
        print(f"  - Max: {max(eigenvector_dict.values()):.8f}")
        print(f"  - Mean: {sum(eigenvector_dict.values())/len(eigenvector_dict):.8f}")
    except:
        print("  Warning: Eigenvector centrality did not converge, using 0 values")
        eigenvector_dict = {node: 0.0 for node in customer_nodes}
    
    # 6. CLUSTERING COEFFICIENT
    print("\n[7] Calculating Clustering Coefficient...")
    print("  (Transitivity - Lecture 3)")
    print("  Projecting bipartite graph to customer-customer network...")
    
    # Project bipartite graph to customer-customer network
    # 2 customers are connected if they bought the same product
    from collections import defaultdict
    
    # Build customer-customer edges
    product_to_customers = defaultdict(set)
    for node in G.nodes():
        if node.startswith('P_'):
            # Get all customers connected to this product
            customers = [n for n in G.neighbors(node) if n.startswith('C_')]
            product_to_customers[node] = set(customers)
    
    # Create customer-customer graph
    G_customer = nx.Graph()
    G_customer.add_nodes_from(customer_nodes)
    
    # Add edges between customers who share products
    for product, customers in product_to_customers.items():
        customers_list = list(customers)
        for i in range(len(customers_list)):
            for j in range(i+1, len(customers_list)):
                G_customer.add_edge(customers_list[i], customers_list[j])
    
    print(f"  Projected graph: {G_customer.number_of_nodes():,} nodes, {G_customer.number_of_edges():,} edges")
    print("  Calculating clustering coefficients...")
    
    clustering_dict_full = nx.clustering(G_customer)
    clustering_dict = {node: clustering_dict_full[node] for node in customer_nodes}
    
    print(f"  Calculated Clustering Coefficient for {len(clustering_dict):,} customers")
    print(f"  - Min: {min(clustering_dict.values()):.6f}")
    print(f"  - Max: {max(clustering_dict.values()):.6f}")
    print(f"  - Mean: {sum(clustering_dict.values())/len(clustering_dict):.6f}")
    
    # 7. COMMUNITY DETECTION
    print("\n[8] Detecting Communities...")
    print("  (Detecting groups of tightly connected nodes)")
    
    try:
        import community as community_louvain
        
        # Louvain algorithm requires undirected graph (already have it)
        print("  Running Louvain algorithm...")
        communities = community_louvain.best_partition(G)
        
        # Extract only customers
        community_dict = {node: communities[node] for node in customer_nodes}
        
        num_communities = len(set(community_dict.values()))
        modularity = community_louvain.modularity(communities, G)
        
        print(f"  Detected {num_communities} communities")
        print(f"  - Modularity score: {modularity:.4f}")
        
        # Community distribution
        from collections import Counter
        comm_counts = Counter(community_dict.values())
        print(f"  - Largest community: {max(comm_counts.values()):,} members")
        print(f"  - Smallest community: {min(comm_counts.values()):,} members")
        
    except ImportError:
        print("  Warning: python-louvain not installed")
        print("  Creating community IDs based on connected components instead...")
        
        community_dict = {}
        for i, component in enumerate(nx.connected_components(G)):
            for node in component:
                if node in customer_nodes:
                    community_dict[node] = i
        
        num_communities = len(set(community_dict.values()))
        print(f"  Created {num_communities} communities from connected components")
    
    # Aggregate results
    print("\n[9] Creating summary DataFrame...")
    
    # Create DataFrame
    results = []
    for node in customer_nodes:
        customer_id = node.replace('C_', '')
        
        results.append({
            'customer_id': customer_id,
            'degree_centrality': degree_dict.get(node, 0),
            'betweenness_centrality': betweenness_dict.get(node, 0),
            'closeness_centrality': closeness_dict.get(node, 0),
            'pagerank': pagerank_dict.get(node, 0),
            'eigenvector_centrality': eigenvector_dict.get(node, 0),
            'clustering_coefficient': clustering_dict.get(node, 0),
            'community_id': community_dict.get(node, 0),
            'degree': G.degree(node),  # Actual degree (number of products)
            'is_fraud': G.nodes[node].get('is_fraud', 0)
        })

    
    df_features = pd.DataFrame(results)
    
    print(f"  Created DataFrame with {len(df_features):,} rows and {len(df_features.columns)} columns")
    
    # Statistics
    print("\n[10] Network features statistics:")
    print(df_features.describe())
    
    # Compare fraud vs normal
    print("\n[11] Comparing Fraud vs Normal customers:")
    
    fraud_df = df_features[df_features['is_fraud'] == 1]
    normal_df = df_features[df_features['is_fraud'] == 0]
    
    print(f"\n  Fraud customers ({len(fraud_df):,}):")
    print(f"    - Avg degree: {fraud_df['degree'].mean():.2f}")
    print(f"    - Avg degree centrality: {fraud_df['degree_centrality'].mean():.6f}")
    print(f"    - Avg betweenness: {fraud_df['betweenness_centrality'].mean():.6f}")
    print(f"    - Avg closeness: {fraud_df['closeness_centrality'].mean():.6f}")
    print(f"    - Avg PageRank: {fraud_df['pagerank'].mean():.8f}")
    print(f"    - Avg eigenvector: {fraud_df['eigenvector_centrality'].mean():.8f}")
    print(f"    - Avg clustering: {fraud_df['clustering_coefficient'].mean():.6f}")
    
    print(f"\n  Normal customers ({len(normal_df):,}):")
    print(f"    - Avg degree: {normal_df['degree'].mean():.2f}")
    print(f"    - Avg degree centrality: {normal_df['degree_centrality'].mean():.6f}")
    print(f"    - Avg betweenness: {normal_df['betweenness_centrality'].mean():.6f}")
    print(f"    - Avg closeness: {normal_df['closeness_centrality'].mean():.6f}")
    print(f"    - Avg PageRank: {normal_df['pagerank'].mean():.8f}")
    print(f"    - Avg eigenvector: {normal_df['eigenvector_centrality'].mean():.8f}")
    print(f"    - Avg clustering: {normal_df['clustering_coefficient'].mean():.6f}")

    
    # Save dictionaries
    print("\n[12] Saving dictionaries...")
    
    features_dict = {
        'degree_centrality': degree_dict,
        'betweenness_centrality': betweenness_dict,
        'closeness_centrality': closeness_dict,
        'pagerank': pagerank_dict,
        'eigenvector_centrality': eigenvector_dict,
        'clustering_coefficient': clustering_dict,
        'community_id': community_dict
    }

    
    with open('data/network_features_dict.pkl', 'wb') as f:
        pickle.dump(features_dict, f)
    print(f"  Saved dictionaries to: data/network_features_dict.pkl")
    
    # Save DataFrame
    df_features.to_csv('data/network_features.csv', index=False)
    print(f"  Saved DataFrame to: data/network_features.csv")
    
    # Summary
    print("\n" + "="*80)
    print("NETWORK FEATURES SUMMARY")
    print("="*80)
    print(f"Calculated 7 types of features:")
    print(f"  1. Degree Centrality - Number of connections")
    print(f"  2. Betweenness Centrality - Bridge role")
    print(f"  3. Closeness Centrality - Distance to other nodes")
    print(f"  4. PageRank - Random walk with teleportation")
    print(f"  5. Eigenvector Centrality - Recursive importance")
    print(f"  6. Clustering Coefficient - Transitivity (fraud rings)")
    print(f"  7. Community ID - Community groups")
    print(f"\nResults:")
    print(f"  - {len(df_features):,} customers with features")
    print(f"  - {num_communities} communities detected")
    print(f"  - Files created:")
    print(f"    * network_features_dict.pkl (7 dictionaries)")
    print(f"    * network_features.csv (DataFrame)")
    print(f"\nReady to compare with traditional features!")
    print("="*80)

    
    return df_features, features_dict


if __name__ == "__main__":
    df_features, features_dict = calculate_network_features()
