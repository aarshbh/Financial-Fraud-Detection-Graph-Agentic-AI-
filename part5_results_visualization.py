"""
part5_results_visualization.py — Final Results & Evaluation
============================================================
PART 5 of the project.

Generates:
• Graph with fraud highlighted (dark theme)
• ROC curves for each model
• Precision-Recall curves
• Feature importance bar chart
• Side-by-side node comparison (fraud vs normal)
• Complete metrics table
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
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import seaborn as sns

from sklearn.metrics import (
    roc_curve, auc as sklearn_auc,
    precision_recall_curve,
    average_precision_score,
    classification_report,
)
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

import config

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def load_all():
    graph_path   = os.path.join(config.DATA_DIR, "graph.pkl")
    results_path = os.path.join(config.DATA_DIR, "detection_results.csv")
    if not os.path.exists(results_path):
        raise FileNotFoundError("Run part3_fraud_detection.py first.")
    with open(graph_path, "rb") as f:
        G, accounts_df, transactions_df = pickle.load(f)
    results_df = pd.read_csv(results_path)
    return G, results_df


# ─────────────────────────────────────────────────────────────────────────────
# 1.  DARK-THEME GRAPH WITH FRAUD HIGHLIGHTED
# ─────────────────────────────────────────────────────────────────────────────

def plot_fraud_graph(G: nx.DiGraph, results_df: pd.DataFrame,
                     max_nodes: int = 150) -> None:
    """
    Visualise the graph with:
    • Red glow  → confirmed fraudulent nodes
    • Orange    → nodes flagged by ensemble (potential fraud)
    • Blue      → normal nodes
    Edge opacity scales with transaction amount.
    """
    sampled = list(G.nodes())[:max_nodes]
    H = G.subgraph(sampled).copy()

    fraud_true = set(
        results_df[results_df["is_fraud"] == 1]["account_id"].tolist()
    )
    fraud_pred = set(
        results_df[results_df["pred_ensemble"] == 1]["account_id"].tolist()
    )

    node_colors, node_sizes, edge_alphas = [], [], []

    for node in H.nodes():
        if node in fraud_true:
            node_colors.append("#FF1744")
            node_sizes.append(280)
        elif node in fraud_pred:
            node_colors.append("#FF9800")
            node_sizes.append(180)
        else:
            node_colors.append("#1565C0")
            node_sizes.append(50)

    amounts = [H[u][v].get("amount", 1) for u, v in H.edges()]
    max_amt = max(amounts) if amounts else 1
    edge_widths = [0.2 + 3.0 * a / max_amt for a in amounts]

    fig, ax = plt.subplots(figsize=(18, 14))
    fig.patch.set_facecolor("#050510")
    ax.set_facecolor("#050510")

    pos = nx.spring_layout(H, seed=config.RANDOM_SEED, k=0.5)

    # Draw edges
    nx.draw_networkx_edges(
        H, pos, ax=ax, alpha=0.15, edge_color="#6888aa",
        width=edge_widths, arrows=True, arrowsize=6,
    )

    # Draw normal nodes first (bottom layer)
    normal_nodes = [n for n in H.nodes() if n not in fraud_true and n not in fraud_pred]
    nx.draw_networkx_nodes(H, pos, nodelist=normal_nodes, ax=ax,
                           node_color="#1565C0", node_size=50, alpha=0.6)

    # Flagged by model (orange)
    flagged_nodes = [n for n in H.nodes() if n in fraud_pred and n not in fraud_true]
    nx.draw_networkx_nodes(H, pos, nodelist=flagged_nodes, ax=ax,
                           node_color="#FF9800", node_size=180, alpha=0.85)

    # True fraud (red, glowing effect via multiple layers)
    true_fraud_nodes = [n for n in H.nodes() if n in fraud_true]
    for size, alpha in [(600, 0.05), (400, 0.1), (280, 0.9)]:
        nx.draw_networkx_nodes(H, pos, nodelist=true_fraud_nodes, ax=ax,
                               node_color="#FF1744", node_size=size, alpha=alpha)

    # Legend
    patches = [
        mpatches.Patch(color="#FF1744", label="True Fraud (confirmed)"),
        mpatches.Patch(color="#FF9800", label="Flagged by Model"),
        mpatches.Patch(color="#1565C0", label="Normal Account"),
    ]
    ax.legend(handles=patches, loc="upper left", fontsize=12,
              facecolor="#0d0d2a", labelcolor="white", framealpha=0.8)

    ax.set_title(
        "Financial Transaction Network — Fraud Detection Results\n"
        f"(Sample: {H.number_of_nodes()} nodes, {H.number_of_edges()} edges)",
        color="white", fontsize=14, fontweight="bold", pad=20,
    )
    ax.axis("off")

    out = os.path.join(config.FIGURES_DIR, "08_fraud_graph.png")
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"[PLOT]  Fraud graph → {out}")
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# 2.  ROC CURVES
# ─────────────────────────────────────────────────────────────────────────────

def plot_roc_curves(results_df: pd.DataFrame) -> None:
    y_true = results_df["is_fraud"].values
    model_score_cols = {
        "Z-Score"           : "pred_zscore",
        "DBSCAN"            : "pred_dbscan",
        "K-Means"           : "pred_kmeans",
        "Isolation Forest"  : "pred_isoforest",
        "Graph-Based Score" : "anomaly_score",
        "Ensemble"          : "pred_ensemble",
    }
    colors = ["#2196F3", "#4CAF50", "#FF9800", "#9C27B0", "#F44336", "#00BCD4"]

    fig, ax = plt.subplots(figsize=(9, 7))
    for (name, col), color in zip(model_score_cols.items(), colors):
        if col not in results_df.columns:
            continue
        scores = results_df[col].fillna(0).values
        fpr, tpr, _ = roc_curve(y_true, scores)
        roc_auc     = sklearn_auc(fpr, tpr)
        ax.plot(fpr, tpr, label=f"{name} (AUC={roc_auc:.3f})",
                color=color, linewidth=2)

    ax.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.5, label="Random")
    ax.fill_between([0, 1], [0, 1], alpha=0.03, color="gray")
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate (Recall)", fontsize=12)
    ax.set_title("ROC Curves — All Models", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", fontsize=10)
    ax.set_xlim([-0.01, 1.01]); ax.set_ylim([-0.01, 1.05])
    ax.grid(True, alpha=0.3)

    out = os.path.join(config.FIGURES_DIR, "09_roc_curves.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"[PLOT]  ROC curves → {out}")
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# 3.  PRECISION-RECALL CURVES
# ─────────────────────────────────────────────────────────────────────────────

def plot_pr_curves(results_df: pd.DataFrame) -> None:
    y_true = results_df["is_fraud"].values
    model_score_cols = {
        "Isolation Forest"  : "pred_isoforest",
        "Graph-Based Score" : "anomaly_score",
        "Ensemble"          : "pred_ensemble",
    }
    colors = ["#9C27B0", "#F44336", "#00BCD4"]

    fig, ax = plt.subplots(figsize=(8, 6))
    for (name, col), color in zip(model_score_cols.items(), colors):
        if col not in results_df.columns:
            continue
        scores = results_df[col].fillna(0).values
        prec, rec, _ = precision_recall_curve(y_true, scores)
        ap = average_precision_score(y_true, scores)
        ax.step(rec, prec, label=f"{name} (AP={ap:.3f})",
                color=color, linewidth=2, where="post")

    baseline = y_true.mean()
    ax.axhline(baseline, ls="--", color="gray", alpha=0.7,
               label=f"Baseline (prevalence={baseline:.3f})")
    ax.set_xlabel("Recall", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_title("Precision-Recall Curves", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1.01]); ax.set_ylim([0, 1.05])

    out = os.path.join(config.FIGURES_DIR, "10_pr_curves.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"[PLOT]  PR curves → {out}")
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# 4.  FEATURE IMPORTANCE (PERMUTATION-STYLE)
# ─────────────────────────────────────────────────────────────────────────────

def plot_feature_importance(results_df: pd.DataFrame) -> None:
    """Rank features by their correlation with the true fraud label."""
    feature_cols = [
        "degree", "in_degree", "out_degree", "in_out_ratio",
        "clustering_coef", "betweenness_centrality", "pagerank",
        "avg_tx_amount", "max_tx_amount", "total_tx_volume",
    ]
    available = [c for c in feature_cols if c in results_df.columns]
    corrs = results_df[available + ["is_fraud"]].corr()["is_fraud"].drop("is_fraud")
    corrs = corrs.abs().sort_values(ascending=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(corrs.index, corrs.values,
                   color=plt.cm.RdYlGn(corrs.values / corrs.max()))
    ax.set_xlabel("|Pearson Correlation| with Fraud Label", fontsize=11)
    ax.set_title("Feature Importance (Correlation with Fraud)",
                 fontsize=13, fontweight="bold")
    for bar, val in zip(bars, corrs.values):
        ax.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", fontsize=9)
    plt.tight_layout()

    out = os.path.join(config.FIGURES_DIR, "11_feature_importance.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"[PLOT]  Feature importance → {out}")
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# 5.  FRAUD vs NORMAL NODE PROFILES (RADAR / BOX PLOTS)
# ─────────────────────────────────────────────────────────────────────────────

def plot_fraud_vs_normal(results_df: pd.DataFrame) -> None:
    """Box plots comparing feature distributions."""
    plot_feats = [
        "degree", "pagerank", "avg_tx_amount",
        "betweenness_centrality", "clustering_coef",
    ]
    available = [c for c in plot_feats if c in results_df.columns]

    fig, axes = plt.subplots(1, len(available), figsize=(16, 6))
    fig.suptitle("Fraud vs Normal: Feature Comparison",
                 fontsize=14, fontweight="bold")

    for ax, feat in zip(axes, available):
        data_normal = results_df[results_df["is_fraud"] == 0][feat].clip(
            upper=results_df[feat].quantile(0.98)
        )
        data_fraud = results_df[results_df["is_fraud"] == 1][feat].clip(
            upper=results_df[feat].quantile(0.98)
        )
        bp = ax.boxplot([data_normal, data_fraud],
                        labels=["Normal", "Fraud"],
                        patch_artist=True, notch=True,
                        medianprops=dict(color="white", linewidth=2))
        bp["boxes"][0].set_facecolor("#2196F3")
        bp["boxes"][1].set_facecolor("#F44336")
        ax.set_title(feat.replace("_", "\n").title(), fontsize=9)
        ax.set_ylabel("Value")
        ax.tick_params(labelsize=8)

    plt.tight_layout()
    out = os.path.join(config.FIGURES_DIR, "12_fraud_vs_normal.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"[PLOT]  Fraud vs Normal → {out}")
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# 6.  FINAL METRICS TABLE
# ─────────────────────────────────────────────────────────────────────────────

def print_final_metrics_table(results_df: pd.DataFrame) -> None:
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    y_true = results_df["is_fraud"].values

    model_cols = {
        "Z-Score"          : "pred_zscore",
        "DBSCAN"           : "pred_dbscan",
        "K-Means"          : "pred_kmeans",
        "Isolation Forest" : "pred_isoforest",
        "Graph-Based"      : "pred_graph",
        "Ensemble"         : "pred_ensemble",
    }

    rows = []
    for name, col in model_cols.items():
        if col not in results_df.columns:
            continue
        y_pred = results_df[col].values
        rows.append({
            "Model"    : name,
            "Accuracy" : f"{accuracy_score(y_true, y_pred):.4f}",
            "Precision": f"{precision_score(y_true, y_pred, zero_division=0):.4f}",
            "Recall"   : f"{recall_score(y_true, y_pred, zero_division=0):.4f}",
            "F1-Score" : f"{f1_score(y_true, y_pred, zero_division=0):.4f}",
        })

    df_metrics = pd.DataFrame(rows)
    out_path = os.path.join(config.REPORTS_DIR, "final_metrics.csv")
    df_metrics.to_csv(out_path, index=False)

    print("\n" + "=" * 70)
    print("  FINAL EVALUATION METRICS TABLE")
    print("=" * 70)
    print(df_metrics.to_string(index=False))
    print("=" * 70)
    print(f"\n[SAVE]  Metrics saved → {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    G, results_df = load_all()
    plot_fraud_graph(G, results_df)
    plot_roc_curves(results_df)
    plot_pr_curves(results_df)
    plot_feature_importance(results_df)
    plot_fraud_vs_normal(results_df)
    print_final_metrics_table(results_df)
    print("\n[DONE]  All Part-5 visualizations generated successfully.")
