# -*- coding: utf-8 -*-
"""
main.py -- Master Pipeline Runner
==================================
Run this single file to execute ALL project parts in order.

Usage:
    python main.py              # Full pipeline
    python main.py --part 1    # Only Part 1 (dataset)
    python main.py --part 2    # Only Part 2 (features)
    python main.py --part 3    # Only Part 3 (detection)
    python main.py --part 4    # Only Part 4 (agentic AI)
    python main.py --part 5    # Only Part 5 (results)
"""

import sys
import time
import argparse
import traceback


def banner(text: str, char: str = "=") -> None:
    width = 65
    print("\n" + char * width)
    print(f"  {text}")
    print(char * width + "\n")


def run_part1():
    banner("PART 1 — Dataset Generation & Graph Construction")
    from part1_dataset import load_dataset, preprocess, build_graph, eda_plots
    import pickle, os, config

    accounts_df, transactions_df = load_dataset()
    accounts_df, transactions_df = preprocess(accounts_df, transactions_df)
    G = build_graph(accounts_df, transactions_df)
    eda_plots(accounts_df, transactions_df)

    graph_path = os.path.join(config.DATA_DIR, "graph.pkl")
    with open(graph_path, "wb") as f:
        pickle.dump((G, accounts_df, transactions_df), f)
    print(f"\n✔  Graph saved → {graph_path}")


def run_part2():
    banner("PART 2 — Graph Feature Engineering & Visualization")
    from part2_graph_features import (
        load_graph, extract_features,
        plot_feature_distributions, visualize_graph, plot_correlation_heatmap,
    )

    G, accounts_df, transactions_df = load_graph()
    features_df = extract_features(G, accounts_df)
    plot_feature_distributions(features_df)
    visualize_graph(G, features_df)
    plot_correlation_heatmap(features_df)
    print("\n✔  Part 2 complete.")


def run_part3():
    banner("PART 3 — Fraud Detection Models")
    from part3_fraud_detection import load_data, run_detection
    from part3_fraud_detection import (
        plot_results, plot_confusion_matrices, plot_metrics_comparison,
    )

    df = load_data()
    results, df_out = run_detection(df)
    plot_results(df_out, results)
    plot_confusion_matrices(df_out, results)
    all_metrics = [v["metrics"] for v in results.values()]
    plot_metrics_comparison(all_metrics)
    print("\n✔  Part 3 complete.")


def run_part4():
    banner("PART 4 — Agentic AI System")
    import pickle, os, pandas as pd, config

    graph_path   = os.path.join(config.DATA_DIR, "graph.pkl")
    features_path = os.path.join(config.DATA_DIR, "node_features.csv")
    results_path  = os.path.join(config.DATA_DIR, "detection_results.csv")

    with open(graph_path, "rb") as f:
        G, _, _ = pickle.load(f)

    features_df = pd.read_csv(features_path)
    results_df  = pd.read_csv(results_path)

    from part4_agentic_ai import OrchestratorAgent
    orchestrator = OrchestratorAgent()
    context = {"graph": G, "features_df": features_df, "results_df": results_df}
    final_msg = orchestrator.act(context)
    print("\n✔  Part 4 complete.")


def run_part5():
    banner("PART 5 — Results & Visualization")
    from part5_results_visualization import (
        load_all, plot_fraud_graph, plot_roc_curves, plot_pr_curves,
        plot_feature_importance, plot_fraud_vs_normal, print_final_metrics_table,
    )

    G, results_df = load_all()
    plot_fraud_graph(G, results_df)
    plot_roc_curves(results_df)
    plot_pr_curves(results_df)
    plot_feature_importance(results_df)
    plot_fraud_vs_normal(results_df)
    print_final_metrics_table(results_df)
    print("\n✔  Part 5 complete.")


PARTS = {1: run_part1, 2: run_part2, 3: run_part3, 4: run_part4, 5: run_part5}


def main():
    parser = argparse.ArgumentParser(
        description="Graph Anomaly Detection — Fraud Detection Pipeline"
    )
    parser.add_argument("--part", type=int, choices=[1, 2, 3, 4, 5],
                        help="Run only a specific part (1-5). Default: all parts.")
    args = parser.parse_args()

    print("\n" + "#" * 65)
    print("  GRAPH ANOMALY DETECTION FOR FRAUD IN FINANCIAL NETWORKS")
    print("  Agentic AI + Pattern Recognition + M.Tech/CSE Project")
    print("#" * 65)

    parts_to_run = [args.part] if args.part else [1, 2, 3, 4, 5]

    for part_num in parts_to_run:
        t0 = time.time()
        try:
            PARTS[part_num]()
            elapsed = time.time() - t0
            print(f"  ✔  Part {part_num} finished in {elapsed:.1f}s")
        except Exception as exc:
            print(f"\n  ✘  Part {part_num} failed: {exc}")
            traceback.print_exc()
            if args.part:
                sys.exit(1)

    banner("ALL DONE -- Project pipeline completed successfully!", "*")
    print("  Outputs stored in:")
    print("    data/              -> datasets, features, detection results")
    print("    outputs/figures/   -> all plots (PNG)")
    print("    outputs/reports/   -> final report, metrics CSV, agent logs")


if __name__ == "__main__":
    main()
