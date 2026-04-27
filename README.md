# 🔍 Graph Anomaly Detection for Fraud in Financial Transaction Networks
### Using Agentic AI · Pattern Recognition · Graph Theory

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/NetworkX-3.3-orange?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Scikit--Learn-1.5-green?style=for-the-badge&logo=scikit-learn&logoColor=white"/>
  <img src="https://img.shields.io/badge/Agentic%20AI-Multi--Agent-purple?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge"/>
</p>

<p align="center">
  <b>M.Tech / CSE  Project &nbsp;|&nbsp; Pattern Recognition Subject</b>
</p>

---

## 📌 Core Idea

> **"Financial fraud is not a single-account problem — it's a network problem."**

Traditional fraud detection looks at each transaction in isolation (row by row in a spreadsheet). But real-world fraud — money laundering, smurfing rings, coordinated account takeovers — leaves patterns in the **relationships between accounts**, not just in the transactions themselves.

This project models an entire bank's transaction history as a **directed weighted graph**, extracts structural features from that graph (PageRank, betweenness centrality, clustering coefficient), runs **5 complementary anomaly detection models**, and then feeds the results to a **multi-agent AI system** that autonomously investigates and explains each suspicious account — just like a team of human fraud analysts, but automated.

---

## 🏗️ System Architecture

```
╔══════════════════════════════════════════════════════════════════════╗
║               FINANCIAL TRANSACTION NETWORK                         ║
║   500 Accounts (Nodes)  ←→  2000 Transactions (Directed Edges)      ║
╚══════════════════════╦═══════════════════════════════════════════════╝
                       ║
                       ▼
╔══════════════════════════════════════════════════════════════════════╗
║                  PART 1 · DATA LAYER                                ║
║                                                                      ║
║   Synthetic Generator  ──►  Preprocessor  ──►  NetworkX DiGraph     ║
║   (or real CSV input)        (clean, type)       graph.pkl saved    ║
╚══════════════════════╦═══════════════════════════════════════════════╝
                       ║
                       ▼
╔══════════════════════════════════════════════════════════════════════╗
║               PART 2 · FEATURE ENGINEERING LAYER                    ║
║                                                                      ║
║   ┌─────────────┐  ┌──────────────────┐  ┌─────────────────────┐   ║
║   │   Degree    │  │   Betweenness    │  │      PageRank       │   ║
║   │  (in / out) │  │   Centrality     │  │   (α = 0.85)        │   ║
║   └─────────────┘  └──────────────────┘  └─────────────────────┘   ║
║   ┌─────────────┐  ┌──────────────────┐  ┌─────────────────────┐   ║
║   │ Clustering  │  │  Eigenvector     │  │  Avg / Max / Total  │   ║
║   │ Coefficient │  │  Centrality      │  │  Transaction Amount │   ║
║   └─────────────┘  └──────────────────┘  └─────────────────────┘   ║
║                                                                      ║
║          16 features per node  →  node_features.csv                 ║
╚══════════════════════╦═══════════════════════════════════════════════╝
                       ║
                       ▼
╔══════════════════════════════════════════════════════════════════════╗
║               PART 3 · DETECTION MODEL LAYER                        ║
║                                                                      ║
║  ┌───────────┐ ┌────────┐ ┌─────────┐ ┌───────────────┐ ┌───────┐  ║
║  │  Z-Score  │ │ DBSCAN │ │ K-Means │ │   Isolation   │ │ Graph │  ║
║  │ (stat)    │ │(density│ │(centroid│ │    Forest     │ │ Score │  ║
║  │ outlier   │ │cluster)│ │distance)│ │  (tree path)  │ │(wgtd) │  ║
║  └─────┬─────┘ └───┬────┘ └────┬────┘ └──────┬────────┘ └──┬────┘  ║
║        └───────────┴───────────┴──────────────┴─────────────┘       ║
║                              ▼                                       ║
║                   ENSEMBLE (Majority Vote ≥ 3/5)                    ║
║                   detection_results.csv saved                        ║
╚══════════════════════╦═══════════════════════════════════════════════╝
                       ║
                       ▼
╔══════════════════════════════════════════════════════════════════════╗
║               PART 4 · AGENTIC AI LAYER                             ║
║                                                                      ║
║         ┌──────────────────────────────────┐                        ║
║         │        ORCHESTRATOR AGENT        │                        ║
║         │  Plans · Routes · Aggregates     │                        ║
║         └──────┬──────────────┬────────────┘                        ║
║                │              │              │                       ║
║                ▼              ▼              ▼                       ║
║    ┌────────────────┐ ┌──────────────┐ ┌──────────────────┐         ║
║    │  DATA ANALYZER │ │  ANOMALY     │ │   EXPLANATION    │         ║
║    │     AGENT      │ │  DETECTION   │ │     AGENT        │         ║
║    │                │ │   AGENT      │ │                  │         ║
║    │ Graph stats    │ │ Rank suspects│ │ LLM reasoning    │         ║
║    │ Node profiles  │ │ Score nodes  │ │ WHY is it fraud? │         ║
║    └────────────────┘ └──────────────┘ └──────────────────┘         ║
║                              ▼                                       ║
║                    final_report.txt  +  agent logs                  ║
╚══════════════════════╦═══════════════════════════════════════════════╝
                       ║
                       ▼
╔══════════════════════════════════════════════════════════════════════╗
║               PART 5 · RESULTS & EVALUATION LAYER                   ║
║                                                                      ║
║   12 PNG plots:  Graph Map · ROC Curves · PR Curves · Heatmap       ║
║                  Feature Importance · Box Plots · Confusion Matrix  ║
║   CSV report:    final_metrics.csv  (Accuracy / Precision / Recall) ║
╚══════════════════════════════════════════════════════════════════════╝
```

---

## 🧩 Pattern Recognition Concepts Used

| Concept | Where Used | Why It Matters for Fraud |
|---------|-----------|--------------------------|
| **Graph Theory** | NetworkX DiGraph | Captures relational fraud patterns invisible in flat tables |
| **Z-Score / Gaussian Statistics** | Part 3 — Approach 1A | Flags accounts statistically far from normal behavior |
| **Density-Based Clustering (DBSCAN)** | Part 3 — Approach 1B | Noise points = accounts that fit no normal cluster |
| **K-Means + Centroid Distance** | Part 3 — Approach 1C | Outliers far from their cluster center are suspicious |
| **Isolation Forest** | Part 3 — Approach 2 | Anomalies isolated in fewer random tree splits |
| **PageRank** | Part 2 Feature | Recursive influence score — fraud hubs score high |
| **Betweenness Centrality** | Part 2 Feature | Money-relay accounts lie on many shortest paths |
| **Clustering Coefficient** | Part 2 Feature | Fraudsters deliberately keep associates disconnected |
| **Ensemble / Majority Voting** | Part 3 Final | Reduces variance; more robust than any single model |
| **PCA (for visualization)** | Part 3 / Part 5 | Projects 12D feature space to 2D for scatter plots |

---

## 📁 Project Structure

```
Fraud_Detection_In_Financial_Banking_System/
│
├── config.py                    ← Central settings (accounts, thresholds, paths)
├── main.py                      ← Master pipeline runner
│
├── part1_dataset.py             ← Dataset generation, preprocessing, graph build
├── part2_graph_features.py      ← 16 graph feature extraction + 4 plots
├── part3_fraud_detection.py     ← 5 models + ensemble + 3 evaluation plots
├── part4_agentic_ai.py          ← Multi-agent AI system (4 agents + MockLLM)
├── part5_results_visualization.py ← Final 5 plots + metrics table
│
├── requirements.txt             ← Python dependencies
│
├── data/                        ← Auto-created on first run
│   ├── accounts.csv
│   ├── transactions.csv
│   ├── graph.pkl
│   ├── node_features.csv
│   └── detection_results.csv
│
└── outputs/
    ├── figures/                 ← 12 PNG plots (01_eda.png ... 12_fraud_vs_normal.png)
    └── reports/
        ├── final_report.txt     ← Agentic AI audit report
        ├── final_metrics.csv    ← Model performance table
        └── agent_*.log          ← Per-agent activity logs
```

---

## ⚙️ Tech Stack

| Library | Version | Role |
|---------|---------|------|
| `networkx` | 3.3 | Graph construction + centrality algorithms |
| `pandas` | 2.2 | Data loading, cleaning, feature tables |
| `numpy` | 1.26 | Numerical computations, Z-scores |
| `scikit-learn` | 1.5 | DBSCAN, K-Means, Isolation Forest, metrics |
| `scipy` | 1.13 | Statistical Z-score computation |
| `matplotlib` | 3.9 | All graph and chart visualizations |
| `seaborn` | 0.13 | Correlation heatmap and distribution plots |
| `openai` *(optional)* | 1.35 | Real LLM explanations (MockLLM used by default) |

---

## 🚀 Quick Start

### 1. Clone the repository
```bash
git clone https://github.com/aarshbh/graph-fraud-detection-agentic-ai.git
cd graph-fraud-detection-agentic-ai
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the full pipeline
```bash
python main.py
```

### 4. Run individual parts
```bash
python main.py --part 1   # Dataset generation & graph construction
python main.py --part 2   # Feature engineering & visualization
python main.py --part 3   # All fraud detection models
python main.py --part 4   # Agentic AI system
python main.py --part 5   # Results & evaluation plots
```

> **No API key needed.** The system uses a rule-based `MockLLM` by default.  
> To use real GPT reasoning, set `OPENAI_API_KEY` in your environment and set `USE_MOCK_LLM = False` in `config.py`.

---

## 📊 Results

### Model Performance (on 500-node synthetic graph, 8% fraud rate)

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| Z-Score | 90.2% | 0.44 | **0.80** | 0.57 |
| DBSCAN | 58.6% | 0.16 | **1.00** | 0.28 |
| K-Means | 86.4% | 0.15 | 0.15 | 0.15 |
| **Isolation Forest** | **94.0%** | **0.63** | 0.63 | **0.63** |
| Graph-Based Score | 91.6% | 0.48 | 0.48 | 0.48 |
| Ensemble (Majority) | 90.8% | 0.45 | 0.70 | 0.55 |

> ⚠️ In fraud detection, **Recall is the priority metric** — missing a fraudster costs far more than a false alarm.

### Generated Plots
```
01_eda.png                 — Transaction & account distribution (EDA)
02_feature_distributions.png — Fraud vs Normal feature histograms
03_graph_visualization.png — Transaction network (dark theme, fraud in red)
04_correlation_heatmap.png — Feature correlation matrix
05_model_results.png       — PCA scatter: TP/FP/TN/FN per model
06_confusion_matrices.png  — Confusion matrices for all models
07_metrics_comparison.png  — Bar chart: model-wise metric comparison
08_fraud_graph.png         — Final dark-theme graph with glowing fraud nodes
09_roc_curves.png          — ROC curves (AUC for all models)
10_pr_curves.png           — Precision-Recall curves
11_feature_importance.png  — Feature correlation with fraud label
12_fraud_vs_normal.png     — Box plots: fraud vs normal accounts
```

---

## 🤖 Agentic AI Design

The system implements a **4-agent autonomous pipeline** inspired by the ReAct (Reasoning + Acting) pattern:

```
User runs: python main.py --part 4
                    │
                    ▼
          OrchestratorAgent
          ├── Step 1: DataAnalyzerAgent.act()
          │       Input : graph.pkl + node_features.csv
          │       Output: Statistical summary of the network
          │
          ├── Step 2: AnomalyDetectionAgent.act()
          │       Input : detection_results.csv
          │       Output: Ranked list of top-15 suspicious accounts
          │
          ├── Step 3: ExplanationAgent.act()
          │       Input : Profiles of top-8 suspects
          │       Output: Natural language explanation per account
          │
          └── Compiles → final_report.txt
```

Each agent communicates via a structured `AgentMessage` protocol carrying `sender`, `receiver`, `content`, `msg_type`, and `timestamp` — enabling full audit trail logging.

---

## 🎓 Academic Context

| Field | Concept Applied |
|-------|----------------|
| **Pattern Recognition** | Z-score, DBSCAN, K-Means, Isolation Forest, PCA |
| **Graph Theory** | PageRank, Betweenness, Eigenvector Centrality, Clustering Coeff |
| **Machine Learning** | Unsupervised anomaly detection (no labels required at inference) |
| **Agentic AI** | Multi-agent orchestration, LLM-based reasoning, tool use |
| **Network Science** | Community structure, hub detection, motif analysis |

**Real-world datasets this system is compatible with:**
- [Elliptic Bitcoin Dataset](https://www.kaggle.com/datasets/ellipticco/elliptic-data-set) — 200K BTC transactions, node labels
- [IEEE-CIS Fraud Detection](https://www.kaggle.com/c/ieee-fraud-detection) — 590K transactions
- [PaySim Mobile Money](https://www.kaggle.com/datasets/ealaxi/paysim1) — 6M synthetic transactions

---

## 📄 License

This project is licensed under the **MIT License** — free to use for academic and personal projects.

---

## 👤 Author

**M.Tech CSE — Final Year Project**  
*Subject: Pattern Recognition*  
*Topic: Graph Anomaly Detection for Fraud Detection in Financial Transaction Networks using Agentic AI*

---

<p align="center">
  Made with Python · NetworkX · Scikit-learn · Matplotlib · Agentic AI
</p>
