"""
part3_fraud_detection.py — Multi-Model Fraud Detection
=======================================================
PART 3 of the project.

THREE COMPLEMENTARY APPROACHES
--------------------------------
Approach 1 — Statistical  : Z-score outlier detection + DBSCAN clustering
Approach 2 — ML           : Isolation Forest (unsupervised; no labels needed)
Approach 3 — Graph-based  : Combining graph features into a composite anomaly score

All three produce a binary prediction per node which are then ensembled.
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
from scipy import stats

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN, KMeans
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, precision_recall_curve, roc_curve,
    accuracy_score, precision_score, recall_score, f1_score,
)
from sklearn.decomposition import PCA

import config

FEATURE_COLS = [
    "degree", "in_degree", "out_degree", "in_out_ratio",
    "clustering_coef", "betweenness_centrality", "pagerank",
    "eigenvector_centrality", "avg_tx_amount", "max_tx_amount",
    "total_tx_volume", "degree_z_score",
]


# ─────────────────────────────────────────────────────────────────────────────
# UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

def load_data() -> pd.DataFrame:
    path = os.path.join(config.DATA_DIR, "node_features.csv")
    if not os.path.exists(path):
        raise FileNotFoundError("Run part2_graph_features.py first.")
    df = pd.read_csv(path)
    print(f"[LOAD]  {len(df)} nodes, {len(df.columns)} columns loaded.")
    return df


def get_X_y(df: pd.DataFrame):
    X = df[FEATURE_COLS].fillna(0).values
    y = df["is_fraud"].values
    return X, y


def print_metrics(name: str, y_true, y_pred, y_score=None) -> dict:
    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec  = recall_score(y_true, y_pred, zero_division=0)
    f1   = f1_score(y_true, y_pred, zero_division=0)
    auc  = roc_auc_score(y_true, y_score) if y_score is not None else None

    print(f"\n{'─'*55}")
    print(f"  MODEL : {name}")
    print(f"  Accuracy  : {acc:.4f}  ({acc*100:.2f}%)")
    print(f"  Precision : {prec:.4f}")
    print(f"  Recall    : {rec:.4f}")
    print(f"  F1-Score  : {f1:.4f}")
    if auc:
        print(f"  ROC-AUC   : {auc:.4f}")
    print(f"{'─'*55}")
    return {"model": name, "accuracy": acc, "precision": prec,
            "recall": rec, "f1": f1, "roc_auc": auc}


# ─────────────────────────────────────────────────────────────────────────────
# APPROACH 1-A : Z-SCORE OUTLIER DETECTION
# ─────────────────────────────────────────────────────────────────────────────

def zscore_detection(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """
    Pattern Recognition Concept:
    ─────────────────────────────
    Z-score measures how many standard deviations a point is from the mean.
    Z = (x − μ) / σ

    Nodes with |Z| > threshold in ANY key feature are flagged as anomalous.
    This is a univariate outlier test applied to multiple features.
    """
    print("\n[APPROACH 1-A]  Z-Score Outlier Detection")
    score_features = [
        "degree", "avg_tx_amount", "max_tx_amount",
        "betweenness_centrality", "pagerank",
    ]
    z_scores = np.abs(stats.zscore(df[score_features].fillna(0)))
    # A node is anomalous if it exceeds threshold in AT LEAST ONE feature
    z_max    = z_scores.max(axis=1)
    y_pred   = (z_max > config.Z_SCORE_THRESHOLD).astype(int)
    print(f"  Z-threshold = {config.Z_SCORE_THRESHOLD}")
    print(f"  Flagged anomalies: {y_pred.sum()} / {len(y_pred)}")
    return y_pred, z_max


# ─────────────────────────────────────────────────────────────────────────────
# APPROACH 1-B : DBSCAN CLUSTERING
# ─────────────────────────────────────────────────────────────────────────────

def dbscan_detection(X: np.ndarray, scaler: StandardScaler) -> np.ndarray:
    """
    Pattern Recognition Concept:
    ─────────────────────────────
    DBSCAN (Density-Based Spatial Clustering of Applications with Noise):
    • Groups points that are closely packed together.
    • Points in low-density regions are labelled as NOISE (label = -1).
    • In fraud detection, NOISE points are our anomalies.

    Unlike K-Means, DBSCAN does not assume spherical clusters — better for
    the irregular shapes financial fraud patterns typically create.
    """
    print("\n[APPROACH 1-B]  DBSCAN Clustering")
    X_scaled = scaler.transform(X)
    # PCA to 6D for speed
    pca = PCA(n_components=6, random_state=config.RANDOM_SEED)
    X_pca = pca.fit_transform(X_scaled)

    db = DBSCAN(eps=config.DBSCAN_EPS,
                min_samples=config.DBSCAN_MIN_SAMPLES,
                metric="euclidean", n_jobs=-1)
    labels = db.fit_predict(X_pca)
    # Noise == -1 → fraud
    y_pred = (labels == -1).astype(int)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    print(f"  Clusters found: {n_clusters}  |  Noise points (fraud): {y_pred.sum()}")
    return y_pred


# ─────────────────────────────────────────────────────────────────────────────
# APPROACH 1-C : K-MEANS CLUSTERING
# ─────────────────────────────────────────────────────────────────────────────

def kmeans_detection(X: np.ndarray, scaler: StandardScaler) -> tuple[np.ndarray, np.ndarray]:
    """
    Pattern Recognition Concept:
    ─────────────────────────────
    K-Means partitions data into K clusters by minimising within-cluster variance.
    Nodes far from their cluster centroid (high reconstruction error) are anomalies.
    """
    print("\n[APPROACH 1-C]  K-Means Clustering (distance to centroid)")
    X_scaled = scaler.transform(X)
    km = KMeans(n_clusters=config.KMEANS_CLUSTERS,
                random_state=config.RANDOM_SEED, n_init=10)
    km.fit(X_scaled)
    centroids = km.cluster_centers_[km.labels_]
    distances  = np.linalg.norm(X_scaled - centroids, axis=1)
    threshold  = np.percentile(distances, 92)   # top 8% = fraud
    y_pred     = (distances > threshold).astype(int)
    print(f"  Distance threshold (92nd pct): {threshold:.4f}")
    print(f"  Flagged: {y_pred.sum()}")
    return y_pred, distances


# ─────────────────────────────────────────────────────────────────────────────
# APPROACH 2 : ISOLATION FOREST
# ─────────────────────────────────────────────────────────────────────────────

def isolation_forest_detection(
    X: np.ndarray, scaler: StandardScaler
) -> tuple[np.ndarray, np.ndarray]:
    """
    Pattern Recognition Concept:
    ─────────────────────────────
    Isolation Forest anomaly detection:
    • Randomly partitions feature space using decision trees.
    • Anomalies are isolated in fewer splits → shorter average path length.
    • Score = negative average path length; more negative = more anomalous.

    No labels needed (fully unsupervised) — ideal for fraud where labels
    are expensive or delayed.
    """
    print("\n[APPROACH 2]  Isolation Forest")
    X_scaled = scaler.transform(X)
    iso = IsolationForest(
        n_estimators=200,
        contamination=config.ISOLATION_FOREST_CONTAMINATION,
        random_state=config.RANDOM_SEED,
        n_jobs=-1,
    )
    iso.fit(X_scaled)
    # predict: -1=anomaly, 1=normal
    raw_pred  = iso.predict(X_scaled)
    scores    = -iso.score_samples(X_scaled)        # higher = more anomalous
    y_pred    = (raw_pred == -1).astype(int)
    print(f"  Contamination: {config.ISOLATION_FOREST_CONTAMINATION}")
    print(f"  Flagged: {y_pred.sum()}")
    return y_pred, scores


# ─────────────────────────────────────────────────────────────────────────────
# APPROACH 3 : GRAPH-BASED COMPOSITE SCORE
# ─────────────────────────────────────────────────────────────────────────────

def graph_based_detection(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """
    Pattern Recognition Concept:
    ─────────────────────────────
    Combine normalised graph metrics into a single COMPOSITE ANOMALY SCORE.

    Score = w1*degree_z + w2*pagerank_z + w3*betweenness_z + w4*amount_z
                + w5*in_out_ratio_z

    Weights are hand-tuned based on domain knowledge:
    • High degree + high amount + unbalanced in/out ratio → strong fraud signal
    """
    print("\n[APPROACH 3]  Graph-Based Composite Score")
    cols = [
        "degree", "pagerank", "betweenness_centrality",
        "avg_tx_amount", "in_out_ratio",
    ]
    data = df[cols].fillna(0).values
    # Normalise each dimension to [0, 1]
    data_norm = (data - data.min(axis=0)) / (np.ptp(data, axis=0) + 1e-9)

    weights = np.array([0.20, 0.25, 0.15, 0.25, 0.15])
    scores  = data_norm @ weights

    threshold = np.percentile(scores, (1 - config.FRAUD_RATIO) * 100)
    y_pred = (scores > threshold).astype(int)
    print(f"  Composite score threshold (p{(1-config.FRAUD_RATIO)*100:.0f}): {threshold:.4f}")
    print(f"  Flagged: {y_pred.sum()}")
    return y_pred, scores


# ─────────────────────────────────────────────────────────────────────────────
# ENSEMBLE (MAJORITY VOTE)
# ─────────────────────────────────────────────────────────────────────────────

def ensemble_predictions(*predictions) -> np.ndarray:
    """
    Majority vote across multiple detectors.
    A node is flagged if MORE THAN HALF the models flag it.
    """
    stack = np.vstack(predictions).T      # (n_nodes, n_models)
    votes = stack.sum(axis=1)
    n_models = stack.shape[1]
    return (votes > n_models / 2).astype(int)


# ─────────────────────────────────────────────────────────────────────────────
# VISUALIZATIONS
# ─────────────────────────────────────────────────────────────────────────────

def plot_results(df: pd.DataFrame, results: dict) -> None:
    """Side-by-side comparison of model predictions via PCA scatter plot."""
    scaler = StandardScaler()
    X = df[FEATURE_COLS].fillna(0).values
    X_scaled = scaler.fit_transform(X)
    pca = PCA(n_components=2, random_state=config.RANDOM_SEED)
    X_2d = pca.fit_transform(X_scaled)

    model_names = list(results.keys())
    n_models = len(model_names)
    ncols = 3
    nrows = (n_models + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(18, 6 * nrows))
    fig.suptitle("Anomaly Detection Results (PCA 2D Projection)",
                 fontsize=15, fontweight="bold")
    axes = axes.flatten()

    y_true = df["is_fraud"].values

    for i, name in enumerate(model_names):
        ax = axes[i]
        y_pred = results[name]["y_pred"]
        # Four groups: TP, FP, TN, FN
        tp = (y_pred == 1) & (y_true == 1)
        fp = (y_pred == 1) & (y_true == 0)
        tn = (y_pred == 0) & (y_true == 0)
        fn = (y_pred == 0) & (y_true == 1)

        ax.scatter(X_2d[tn, 0], X_2d[tn, 1], s=15, c="#2196F3",  alpha=0.4, label="TN (Normal)")
        ax.scatter(X_2d[fp, 0], X_2d[fp, 1], s=30, c="#FF9800",  alpha=0.7, label="FP (False Alarm)")
        ax.scatter(X_2d[fn, 0], X_2d[fn, 1], s=50, c="#9C27B0",  alpha=0.9, label="FN (Missed)")
        ax.scatter(X_2d[tp, 0], X_2d[tp, 1], s=80, c="#F44336",  alpha=0.9,
                   marker="*", label="TP (Fraud Caught)")

        acc  = results[name]["metrics"]["accuracy"]
        rec  = results[name]["metrics"]["recall"]
        ax.set_title(f"{name}\nAcc={acc:.2%} | Recall={rec:.2%}", fontsize=10)
        ax.legend(fontsize=7, loc="upper right")
        ax.set_xlabel("PC1"); ax.set_ylabel("PC2")

    # Hide unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    out = os.path.join(config.FIGURES_DIR, "05_model_results.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"[PLOT]  Model results → {out}")
    plt.close()


def plot_confusion_matrices(df: pd.DataFrame, results: dict) -> None:
    model_names = list(results.keys())
    n = len(model_names)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5))
    if n == 1:
        axes = [axes]
    fig.suptitle("Confusion Matrices", fontsize=14, fontweight="bold")

    y_true = df["is_fraud"].values
    for ax, name in zip(axes, model_names):
        cm = confusion_matrix(y_true, results[name]["y_pred"])
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                    xticklabels=["Normal", "Fraud"],
                    yticklabels=["Normal", "Fraud"],
                    cbar=False)
        ax.set_title(name, fontsize=11, fontweight="bold")
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")

    plt.tight_layout()
    out = os.path.join(config.FIGURES_DIR, "06_confusion_matrices.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"[PLOT]  Confusion matrices → {out}")
    plt.close()


def plot_metrics_comparison(all_metrics: list[dict]) -> None:
    df_m = pd.DataFrame(all_metrics).set_index("model")
    df_m = df_m.drop(columns=["roc_auc"], errors="ignore")

    ax = df_m.plot(kind="bar", figsize=(14, 6), rot=25,
                   color=["#2196F3", "#4CAF50", "#F44336", "#FF9800"])
    ax.set_title("Model Performance Comparison", fontsize=14, fontweight="bold")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.1)
    ax.axhline(0.9, ls="--", color="gray", alpha=0.5, label="0.90 baseline")
    ax.legend(loc="lower right")
    plt.tight_layout()
    out = os.path.join(config.FIGURES_DIR, "07_metrics_comparison.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"[PLOT]  Metrics comparison → {out}")
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def run_detection(df: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    X, y_true = get_X_y(df)
    scaler = StandardScaler().fit(X)

    results     = {}
    all_metrics = []

    # ── Z-Score ────────────────────────────────────────────────────────────────
    y_z, z_max = zscore_detection(df)
    m = print_metrics("Z-Score", y_true, y_z, z_max)
    results["Z-Score"]     = {"y_pred": y_z, "metrics": m}
    all_metrics.append(m)

    # ── DBSCAN ────────────────────────────────────────────────────────────────
    y_db = dbscan_detection(X, scaler)
    m = print_metrics("DBSCAN", y_true, y_db)
    results["DBSCAN"]      = {"y_pred": y_db, "metrics": m}
    all_metrics.append(m)

    # ── K-Means ───────────────────────────────────────────────────────────────
    y_km, km_dist = kmeans_detection(X, scaler)
    m = print_metrics("K-Means", y_true, y_km, km_dist / km_dist.max())
    results["K-Means"]     = {"y_pred": y_km, "metrics": m}
    all_metrics.append(m)

    # ── Isolation Forest ──────────────────────────────────────────────────────
    y_if, if_scores = isolation_forest_detection(X, scaler)
    m = print_metrics("Isolation Forest", y_true, y_if,
                      if_scores / if_scores.max())
    results["Isolation Forest"] = {"y_pred": y_if, "metrics": m}
    all_metrics.append(m)

    # ── Graph-Based ───────────────────────────────────────────────────────────
    y_gp, gp_scores = graph_based_detection(df)
    m = print_metrics("Graph-Based Score", y_true, y_gp,
                      gp_scores / gp_scores.max())
    results["Graph-Based Score"] = {"y_pred": y_gp, "metrics": m}
    all_metrics.append(m)

    # ── Ensemble ──────────────────────────────────────────────────────────────
    y_ens = ensemble_predictions(y_z, y_db, y_km, y_if, y_gp)
    m = print_metrics("Ensemble (Majority)", y_true, y_ens)
    results["Ensemble (Majority)"] = {"y_pred": y_ens, "metrics": m}
    all_metrics.append(m)

    # Attach predictions to df
    df_out = df.copy()
    df_out["pred_zscore"]     = y_z
    df_out["pred_dbscan"]     = y_db
    df_out["pred_kmeans"]     = y_km
    df_out["pred_isoforest"]  = y_if
    df_out["pred_graph"]      = y_gp
    df_out["pred_ensemble"]   = y_ens
    df_out["anomaly_score"]   = gp_scores     # used by agent

    out_path = os.path.join(config.DATA_DIR, "detection_results.csv")
    df_out.to_csv(out_path, index=False)
    print(f"\n[SAVE]  Detection results → {out_path}")

    return results, df_out


if __name__ == "__main__":
    df = load_data()
    results, df_out = run_detection(df)
    plot_results(df_out, results)
    plot_confusion_matrices(df_out, results)

    all_metrics = [v["metrics"] for v in results.values()]
    plot_metrics_comparison(all_metrics)
