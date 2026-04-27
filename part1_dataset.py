"""
part1_dataset.py — Dataset Generation, Loading, Cleaning & Graph Construction
=============================================================================
PART 1 of the project.

REAL-WORLD DATASET OPTIONS
---------------------------
1. IEEE-CIS Fraud Detection (Kaggle):
   https://www.kaggle.com/c/ieee-fraud-detection
   → Contains 590K transactions with rich features; good for edge-level fraud.

2. Elliptic Bitcoin Dataset (Kaggle / MIT):
   https://www.kaggle.com/datasets/ellipticco/elliptic-data-set
   → 200K Bitcoin transactions as a directed graph; node labels available.

3. PaySim Synthetic Mobile Money (Kaggle):
   https://www.kaggle.com/datasets/ealaxi/paysim1
   → 6M synthetic mobile money transactions with fraud flags.

Since downloading these requires Kaggle authentication, we generate a
SYNTHETIC dataset that mirrors the same statistical properties.
"""

import os
import random
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib
matplotlib.use("Agg")   # non-interactive backend — saves files without blocking
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta

import config

# ─────────────────────────────────────────────────────────────────────────────
# 1.  SYNTHETIC DATASET GENERATION
# ─────────────────────────────────────────────────────────────────────────────

def generate_synthetic_dataset(
    n_accounts: int    = config.N_ACCOUNTS,
    n_transactions: int = config.N_TRANSACTIONS,
    fraud_ratio: float = config.FRAUD_RATIO,
    seed: int          = config.RANDOM_SEED,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generate a realistic synthetic financial transaction dataset.

    Returns
    -------
    accounts_df : pd.DataFrame
        Node table  — one row per bank account.
    transactions_df : pd.DataFrame
        Edge table  — one row per transaction (directed: sender → receiver).
    """
    rng = np.random.default_rng(seed)
    random.seed(seed)

    # ── Accounts (Nodes) ──────────────────────────────────────────────────────
    account_ids   = [f"ACC_{i:04d}" for i in range(n_accounts)]
    account_types = rng.choice(["personal", "business", "corporate"],
                               size=n_accounts, p=[0.6, 0.3, 0.1])
    n_fraud       = int(n_accounts * fraud_ratio)
    is_fraud      = [0] * n_accounts
    fraud_indices = rng.choice(n_accounts, size=n_fraud, replace=False)
    for idx in fraud_indices:
        is_fraud[idx] = 1

    # Account balance: fraudsters tend to have irregular balances
    balances = rng.exponential(scale=10_000, size=n_accounts)
    balances[fraud_indices] *= rng.uniform(3, 10, size=n_fraud)  # inflated

    accounts_df = pd.DataFrame({
        "account_id"   : account_ids,
        "account_type" : account_types,
        "balance"      : np.round(balances, 2),
        "is_fraud"     : is_fraud,
    })

    # ── Transactions (Edges) ──────────────────────────────────────────────────
    fraud_set = set(fraud_indices)

    senders, receivers, amounts, timestamps, tx_types = [], [], [], [], []
    base_time = datetime(2024, 1, 1)

    for _ in range(n_transactions):
        sender_idx = rng.integers(0, n_accounts)
        receiver_idx = rng.integers(0, n_accounts)
        while receiver_idx == sender_idx:
            receiver_idx = rng.integers(0, n_accounts)

        # Fraudsters send larger amounts
        if sender_idx in fraud_set:
            amount = rng.uniform(
                config.MIN_AMOUNT * config.FRAUD_MULTIPLIER,
                config.MAX_AMOUNT * config.FRAUD_MULTIPLIER,
            )
        else:
            amount = rng.exponential(scale=1_500)
            amount = np.clip(amount, config.MIN_AMOUNT, config.MAX_AMOUNT)

        # Random timestamp within 90 days
        delta_seconds = rng.integers(0, 90 * 24 * 3600)
        ts = base_time + timedelta(seconds=int(delta_seconds))

        tx_type = rng.choice(["transfer", "payment", "withdrawal"],
                             p=[0.5, 0.35, 0.15])

        senders.append(account_ids[sender_idx])
        receivers.append(account_ids[receiver_idx])
        amounts.append(round(float(amount), 2))
        timestamps.append(ts.strftime("%Y-%m-%d %H:%M:%S"))
        tx_types.append(tx_type)

    transactions_df = pd.DataFrame({
        "sender"      : senders,
        "receiver"    : receivers,
        "amount"      : amounts,
        "timestamp"   : timestamps,
        "tx_type"     : tx_types,
    })

    print(f"[DATASET]  Accounts   : {len(accounts_df)}  "
          f"(fraud={is_fraud.count(1)}, {fraud_ratio*100:.0f}%)")
    print(f"[DATASET]  Transactions: {len(transactions_df)}")
    return accounts_df, transactions_df


# ─────────────────────────────────────────────────────────────────────────────
# 2.  DATA LOADING  (real CSV  OR  synthetic fallback)
# ─────────────────────────────────────────────────────────────────────────────

def load_dataset(
    accounts_path: str | None = None,
    transactions_path: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load from CSVs if paths are given, otherwise generate synthetic data.
    Saves generated data to data/ for reproducibility.
    """
    acc_file = os.path.join(config.DATA_DIR, "accounts.csv")
    tx_file  = os.path.join(config.DATA_DIR, "transactions.csv")

    if accounts_path and os.path.exists(accounts_path):
        accounts_df     = pd.read_csv(accounts_path)
        transactions_df = pd.read_csv(transactions_path)
        print("[LOAD]  Loaded real-world dataset from CSV.")
    elif os.path.exists(acc_file):
        accounts_df     = pd.read_csv(acc_file)
        transactions_df = pd.read_csv(tx_file)
        print("[LOAD]  Loaded cached synthetic dataset.")
    else:
        accounts_df, transactions_df = generate_synthetic_dataset()
        accounts_df.to_csv(acc_file,  index=False)
        transactions_df.to_csv(tx_file, index=False)
        print(f"[SAVE]  Saved to {config.DATA_DIR}")

    return accounts_df, transactions_df


# ─────────────────────────────────────────────────────────────────────────────
# 3.  DATA CLEANING & PREPROCESSING
# ─────────────────────────────────────────────────────────────────────────────

def preprocess(
    accounts_df: pd.DataFrame,
    transactions_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Clean and preprocess both tables.
    Steps:
    - Drop duplicates
    - Handle missing values
    - Convert types
    - Remove self-loops
    - Feature engineer: hour_of_day, is_large_tx
    """
    print("\n[PREPROCESS] Starting ...")

    # ── Accounts ──────────────────────────────────────────────────────────────
    before = len(accounts_df)
    accounts_df = accounts_df.drop_duplicates(subset=["account_id"])
    accounts_df["balance"] = accounts_df["balance"].fillna(
        accounts_df["balance"].median()
    )
    accounts_df["is_fraud"] = accounts_df["is_fraud"].astype(int)
    print(f"  Accounts  : {before} → {len(accounts_df)} (dupes removed)")

    # ── Transactions ──────────────────────────────────────────────────────────
    before = len(transactions_df)
    transactions_df = transactions_df.drop_duplicates()
    transactions_df = transactions_df.dropna(subset=["sender", "receiver", "amount"])
    transactions_df = transactions_df[
        transactions_df["sender"] != transactions_df["receiver"]   # no self-loops
    ]
    transactions_df["amount"] = transactions_df["amount"].clip(lower=0)
    transactions_df["timestamp"] = pd.to_datetime(transactions_df["timestamp"])
    transactions_df["hour_of_day"] = transactions_df["timestamp"].dt.hour
    q95 = transactions_df["amount"].quantile(0.95)
    transactions_df["is_large_tx"] = (transactions_df["amount"] > q95).astype(int)
    print(f"  Transactions: {before} → {len(transactions_df)} "
          f"(self-loops removed, types fixed)")
    print(f"  Large-tx threshold (95th pct): ${q95:,.0f}")
    print("[PREPROCESS] Done.\n")

    return accounts_df, transactions_df


# ─────────────────────────────────────────────────────────────────────────────
# 4.  GRAPH CONSTRUCTION
# ─────────────────────────────────────────────────────────────────────────────

def build_graph(
    accounts_df: pd.DataFrame,
    transactions_df: pd.DataFrame,
) -> nx.DiGraph:
    """
    Convert the tabular data into a directed weighted graph.

    Nodes  → bank accounts  (attributes: balance, is_fraud, account_type)
    Edges  → transactions   (attributes: amount, timestamp, tx_type)
    """
    G = nx.DiGraph()

    # ── Add nodes ─────────────────────────────────────────────────────────────
    for _, row in accounts_df.iterrows():
        G.add_node(
            row["account_id"],
            balance      = row["balance"],
            is_fraud     = row["is_fraud"],
            account_type = row["account_type"],
        )

    # ── Add edges ─────────────────────────────────────────────────────────────
    for _, row in transactions_df.iterrows():
        src, dst = row["sender"], row["receiver"]
        if G.has_edge(src, dst):
            # Multiple transactions between same pair → accumulate
            G[src][dst]["amount"]  += row["amount"]
            G[src][dst]["tx_count"] += 1
        else:
            G.add_edge(
                src, dst,
                amount    = row["amount"],
                tx_count  = 1,
                tx_type   = row["tx_type"],
                hour      = row["hour_of_day"],
            )

    print(f"[GRAPH]  Nodes: {G.number_of_nodes()}  |  "
          f"Edges: {G.number_of_edges()}")
    return G


# ─────────────────────────────────────────────────────────────────────────────
# 5.  QUICK EDA VISUALIZATIONS
# ─────────────────────────────────────────────────────────────────────────────

def eda_plots(accounts_df: pd.DataFrame, transactions_df: pd.DataFrame) -> None:
    """Generate exploratory data analysis plots."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Exploratory Data Analysis — Financial Transaction Dataset",
                 fontsize=14, fontweight="bold")

    # Plot 1: Fraud vs Normal accounts
    ax = axes[0]
    counts = accounts_df["is_fraud"].value_counts()
    ax.bar(["Normal", "Fraudulent"], counts.values,
           color=["#2196F3", "#F44336"], edgecolor="white", linewidth=1.5)
    ax.set_title("Account Distribution")
    ax.set_ylabel("Count")
    for i, v in enumerate(counts.values):
        ax.text(i, v + 3, str(v), ha="center", fontweight="bold")

    # Plot 2: Transaction amount distribution
    ax = axes[1]
    ax.hist(transactions_df["amount"].clip(upper=50_000),
            bins=50, color="#4CAF50", edgecolor="white", linewidth=0.3)
    ax.set_title("Transaction Amount Distribution")
    ax.set_xlabel("Amount ($)")
    ax.set_ylabel("Frequency")
    ax.set_yscale("log")

    # Plot 3: Transactions by hour
    ax = axes[2]
    hour_counts = transactions_df["hour_of_day"].value_counts().sort_index()
    ax.plot(hour_counts.index, hour_counts.values,
            color="#9C27B0", linewidth=2, marker="o", markersize=4)
    ax.fill_between(hour_counts.index, hour_counts.values, alpha=0.2, color="#9C27B0")
    ax.set_title("Transactions by Hour of Day")
    ax.set_xlabel("Hour")
    ax.set_ylabel("Count")
    ax.set_xticks(range(0, 24, 2))

    plt.tight_layout()
    out = os.path.join(config.FIGURES_DIR, "01_eda.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"[PLOT]  EDA saved → {out}")
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    accounts_df, transactions_df = load_dataset()
    accounts_df, transactions_df = preprocess(accounts_df, transactions_df)
    G = build_graph(accounts_df, transactions_df)
    eda_plots(accounts_df, transactions_df)

    # Save graph for downstream use
    import pickle
    graph_path = os.path.join(config.DATA_DIR, "graph.pkl")
    with open(graph_path, "wb") as f:
        pickle.dump((G, accounts_df, transactions_df), f)
    print(f"[SAVE]  Graph saved → {graph_path}")
