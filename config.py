"""
config.py — Central configuration for all project modules
"""

import os

# ─────────────────────────────────────────────
# RANDOM SEED
# ─────────────────────────────────────────────
RANDOM_SEED = 42

# ─────────────────────────────────────────────
# DATASET SETTINGS
# ─────────────────────────────────────────────
N_ACCOUNTS        = 500        # Total nodes (bank accounts)
N_TRANSACTIONS    = 2000       # Total edges (transactions)
FRAUD_RATIO       = 0.08       # 8% fraudulent accounts
MIN_AMOUNT        = 100        # Min transaction amount ($)
MAX_AMOUNT        = 50_000     # Max transaction amount ($)
FRAUD_MULTIPLIER  = 5          # Fraudsters transact 5× normal amount

# ─────────────────────────────────────────────
# GRAPH SETTINGS
# ─────────────────────────────────────────────
GRAPH_DIRECTED    = True       # Directed graph (sender → receiver)

# ─────────────────────────────────────────────
# MODEL SETTINGS
# ─────────────────────────────────────────────
Z_SCORE_THRESHOLD = 2.5        # Nodes beyond this are anomalous
DBSCAN_EPS        = 0.5
DBSCAN_MIN_SAMPLES = 5
ISOLATION_FOREST_CONTAMINATION = 0.08   # matches FRAUD_RATIO
KMEANS_CLUSTERS   = 3

# ─────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────
BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
DATA_DIR      = os.path.join(BASE_DIR, "data")
OUTPUT_DIR    = os.path.join(BASE_DIR, "outputs")
FIGURES_DIR   = os.path.join(OUTPUT_DIR, "figures")
REPORTS_DIR   = os.path.join(OUTPUT_DIR, "reports")

for _dir in [DATA_DIR, OUTPUT_DIR, FIGURES_DIR, REPORTS_DIR]:
    os.makedirs(_dir, exist_ok=True)

# ─────────────────────────────────────────────
# LLM / AGENTIC AI SETTINGS
# ─────────────────────────────────────────────
# Set your OpenAI key in .env or as an env var.
# Leave blank → the system uses rule-based mock agents (no API needed).
OPENAI_API_KEY    = os.getenv("OPENAI_API_KEY", "")
USE_MOCK_LLM      = True       # Set False only when real API key is present
LLM_MODEL         = "gpt-3.5-turbo"
LLM_TEMPERATURE   = 0.3
