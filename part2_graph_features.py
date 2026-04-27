"""
part2_graph_features.py — Graph Feature Engineering & Visualization
====================================================================
PART 2 of the project.

PATTERN RECOGNITION CONCEPTS USED
-----------------------------------
• Degree          → How connected a node is; highly connected = hub/anomaly
• Clustering Coef → Fraction of neighbor-pairs that are themselves connected
                    (low = possible money-mule or isolated fraudster)
• Betweenness     → Nodes on many shortest paths = chokepoints (bridges)
• PageRank        → Recursive prestige — fraudsters often have high PR
• In/Out-degree   → Senders vs receivers; fraud accounts often skew heavily
"""

import os
import pickle
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib
matplotlib.use("Agg")  # non-interactive: save to file
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from collections import defaultdict

import config


# ─────────────────────────────────────────────────────────────────────────────
# 1.  LOAD GRAPH
# ─────────────────────────────────────────────────────────────────────────────

def load_graph() -> tuple[nx.DiGraph, pd.DataFrame, pd.DataFrame]:
    path = os.path.join(config.DATA_DIR, "graph.pkl")
    if not os.path.exists(path):
        raise FileNotFoundError(
            "Run part1_dataset.py first to generate graph.pkl"
        )
    with open(path, "rb") as f:
        G, accounts_df, transactions_df = pickle.load(f)
    print(f"[LOAD]  Graph loaded — {G.number_of_nodes()} nodes, "
          f"{G.number_of_edges()} edges")
    return G, accounts_df, transactions_df


# ─────────────────────────────────────────────────────────────────────────────
# 2.  FEATURE EXTRACTION
# ─────────────────────────────────────────────────────────────────────────────

def extract_features(G: nx.DiGraph, accounts_df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract graph-based features for every node.

    Feature                  Pattern Recognition Meaning
    ─────────────────────────────────────────────────────────────────────────
    degree                   Total connections  (in + out)
    in_degree                Number of incoming transactions
    out_degree               Number of outgoing transactions
    in_out_ratio             Imbalance indicator; fraud often heavily one-sided
    clustering_coef          Local density; low → isolated fraudster
    betweenness_centrality   Chokepoint importance
    pagerank                 Recursive prestige / influence
    eigenvector_centrality   Influence based on neighbor influence
    avg_tx_amount            Mean transaction amount on outgoing edges
    max_tx_amount            Max single transaction (spike detection)
    total_tx_volume          Total flow through the node
    degree_z_score           Standardized degree (for outlier detection)
    """
    print("[FEATURES] Extracting graph features …")

    nodes = list(G.nodes())

    # Basic degree measures
    deg          = dict(G.degree())
    in_deg       = dict(G.in_degree())
    out_deg      = dict(G.out_degree())

    # Graph-theoretic centrality
    print("  Computing betweenness centrality (may take a moment) …")
    betweenness  = nx.betweenness_centrality(G, normalized=True, weight="amount")
    pagerank     = nx.pagerank(G, alpha=0.85, weight="amount")

    try:
        eigenvector = nx.eigenvector_centrality(G, max_iter=500, weight="amount")
    except nx.PowerIterationFailedConvergence:
        eigenvector = {n: 0.0 for n in nodes}

    # Clustering coefficient (undirected version for speed)
    G_undirected = G.to_undirected()
    clustering   = nx.clustering(G_undirected)

    # Transaction-level aggregates per node (outgoing edges)
    avg_amount = {}
    max_amount = {}
    total_vol  = {}
    for node in nodes:
        amounts = [
            G[node][nbr].get("amount", 0)
            for nbr in G.successors(node)
        ]
        avg_amount[node] = np.mean(amounts) if amounts else 0.0
        max_amount[node] = np.max(amounts)  if amounts else 0.0
        total_vol[node]  = np.sum(amounts)  if amounts else 0.0

    # Assemble feature table
    feature_rows = []
    for node in nodes:
        in_d  = in_deg[node]
        out_d = out_deg[node]
        ratio = out_d / (in_d + 1e-9)  # avoid div/0
        feature_rows.append({
            "account_id"             : node,
            "degree"                 : deg[node],
            "in_degree"              : in_d,
            "out_degree"             : out_d,
            "in_out_ratio"           : round(ratio, 4),
            "clustering_coef"        : round(clustering.get(node, 0.0), 4),
            "betweenness_centrality" : round(betweenness.get(node, 0.0), 6),
            "pagerank"               : round(pagerank.get(node, 0.0), 6),
            "eigenvector_centrality" : round(eigenvector.get(node, 0.0), 6),
            "avg_tx_amount"          : round(avg_amount[node], 2),
            "max_tx_amount"          : round(max_amount[node], 2),
            "total_tx_volume"        : round(total_vol[node], 2),
        })

    features_df = pd.DataFrame(feature_rows)

    # Merge ground-truth labels
    features_df = features_df.merge(
        accounts_df[["account_id", "is_fraud", "balance", "account_type"]],
        on="account_id", how="left"
    )

    # Z-score of degree (useful for statistical anomaly detection)
    mu  = features_df["degree"].mean()
    std = features_df["degree"].std()
    features_df["degree_z_score"] = ((features_df["degree"] - mu) / (std + 1e-9)).round(4)

    # Save
    out_path = os.path.join(config.DATA_DIR, "node_features.csv")
    features_df.to_csv(out_path, index=False)
    print(f"[FEATURES] Done — {features_df.shape[1]} features for "
          f"{len(features_df)} nodes → {out_path}")
    return features_df


# ─────────────────────────────────────────────────────────────────────────────
# 3.  FEATURE DISTRIBUTION PLOTS
# ─────────────────────────────────────────────────────────────────────────────

def plot_feature_distributions(features_df: pd.DataFrame) -> None:
    """Compare feature distributions: Fraud vs Normal."""
    plot_features = [
        "degree", "betweenness_centrality", "pagerank",
        "avg_tx_amount", "clustering_coef", "in_out_ratio",
    ]
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("Feature Distributions: Fraudulent vs Normal Accounts",
                 fontsize=14, fontweight="bold")
    axes = axes.flatten()

    fraud_df  = features_df[features_df["is_fraud"] == 1]
    normal_df = features_df[features_df["is_fraud"] == 0]

    for ax, feat in zip(axes, plot_features):
        ax.hist(normal_df[feat].clip(upper=normal_df[feat].quantile(0.99)),
                bins=40, alpha=0.6, color="#2196F3", label="Normal", density=True)
        ax.hist(fraud_df[feat].clip(upper=fraud_df[feat].quantile(0.99)),
                bins=40, alpha=0.7, color="#F44336", label="Fraud", density=True)
        ax.set_title(feat.replace("_", " ").title())
        ax.set_ylabel("Density")
        ax.legend()

    plt.tight_layout()
    out = os.path.join(config.FIGURES_DIR, "02_feature_distributions.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"[PLOT]  Feature distributions → {out}")
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# 4.  GRAPH VISUALIZATION
# ─────────────────────────────────────────────────────────────────────────────

def visualize_graph(G: nx.DiGraph, features_df: pd.DataFrame,
                    max_nodes: int = 120) -> None:
    """
    Draw a subgraph highlighting fraudulent nodes in red.
    We sample 'max_nodes' for visual clarity.
    """
    # Sample for clarity
    sampled_nodes = list(G.nodes())[:max_nodes]
    H = G.subgraph(sampled_nodes).copy()

    # Node color: red = fraud, blue = normal
    fraud_set = set(
        features_df[features_df["is_fraud"] == 1]["account_id"].tolist()
    )
    node_colors = []
    node_sizes  = []
    for node in H.nodes():
        if node in fraud_set:
            node_colors.append("#F44336")
            node_sizes.append(200)
        else:
            node_colors.append("#2196F3")
            node_sizes.append(60)

    # Edge widths proportional to amount
    max_amt = max((H[u][v].get("amount", 1) for u, v in H.edges()), default=1)
    edge_widths = [
        0.3 + 2 * H[u][v].get("amount", 0) / max_amt
        for u, v in H.edges()
    ]

    fig, ax = plt.subplots(figsize=(16, 12))
    fig.patch.set_facecolor("#0d0d1a")
    ax.set_facecolor("#0d0d1a")

    pos = nx.spring_layout(H, seed=config.RANDOM_SEED, k=0.4)

    nx.draw_networkx_edges(H, pos, ax=ax, alpha=0.25,
                           edge_color="#aaaaaa", width=edge_widths,
                           arrows=True, arrowsize=8)
    nx.draw_networkx_nodes(H, pos, ax=ax,
                           node_color=node_colors, node_size=node_sizes,
                           alpha=0.9)

    # Legend
    normal_patch = mpatches.Patch(color="#2196F3", label="Normal Account")
    fraud_patch  = mpatches.Patch(color="#F44336", label="Fraudulent Account")
    ax.legend(handles=[normal_patch, fraud_patch], loc="upper left",
              facecolor="#1a1a2e", labelcolor="white", fontsize=12)

    ax.set_title(
        f"Financial Transaction Network (sample: {H.number_of_nodes()} nodes, "
        f"{H.number_of_edges()} edges)",
        color="white", fontsize=14, fontweight="bold", pad=15
    )
    ax.axis("off")

    out = os.path.join(config.FIGURES_DIR, "03_graph_visualization.png")
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"[PLOT]  Graph visualization → {out}")
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# 5.  CORRELATION HEATMAP
# ─────────────────────────────────────────────────────────────────────────────

def plot_correlation_heatmap(features_df: pd.DataFrame) -> None:
    numeric_cols = [
        "degree", "in_degree", "out_degree", "clustering_coef",
        "betweenness_centrality", "pagerank", "avg_tx_amount",
        "max_tx_amount", "total_tx_volume", "is_fraud",
    ]
    corr = features_df[numeric_cols].corr()

    fig, ax = plt.subplots(figsize=(12, 9))
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm",
                ax=ax, linewidths=0.5, mask=~mask,
                vmin=-1, vmax=1, square=True, cbar_kws={"shrink": 0.8})
    ax.set_title("Feature Correlation Heatmap", fontsize=14, fontweight="bold")
    plt.tight_layout()
    out = os.path.join(config.FIGURES_DIR, "04_correlation_heatmap.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"[PLOT]  Correlation heatmap → {out}")
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    G, accounts_df, transactions_df = load_graph()
    features_df = extract_features(G, accounts_df)

    print("\nFeature summary (first 5 rows):")
    print(features_df.head())
    print("\nFeature statistics:")
    print(features_df.describe().round(3))

    plot_feature_distributions(features_df)
    visualize_graph(G, features_df)
    plot_correlation_heatmap(features_df)
