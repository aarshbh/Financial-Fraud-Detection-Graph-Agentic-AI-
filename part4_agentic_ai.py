"""
part4_agentic_ai.py — Agentic AI System for Fraud Detection
============================================================
PART 4 of the project.

ARCHITECTURE
────────────────────────────────────────────────────────────────────────────
                    ┌───────────────────────────────────┐
                    │        ORCHESTRATOR AGENT         │
                    │  (Coordinates all sub-agents,     │
                    │   routes tasks, collects results) │
                    └──────────────┬────────────────────┘
                                   │
            ┌──────────────────────┼──────────────────────┐
            ▼                      ▼                       ▼
  ┌─────────────────┐   ┌──────────────────────┐  ┌──────────────────────┐
  │  DATA ANALYZER  │   │  ANOMALY DETECTION   │  │  EXPLANATION AGENT   │
  │     AGENT       │   │       AGENT          │  │                      │
  │                 │   │                      │  │  WHY is a node       │
  │ • Graph stats   │   │ • Runs detectors     │  │  fraudulent?         │
  │ • Node profile  │   │ • Computes scores    │  │  Human-readable      │
  │ • Risk factors  │   │ • Flags anomalies    │  │  reasoning           │
  └─────────────────┘   └──────────────────────┘  └──────────────────────┘

IMPLEMENTATION NOTE
────────────────────
The system uses a MOCK LLM (rule-based) by default so it runs WITHOUT
any API key or internet connection.
Set USE_MOCK_LLM = False in config.py and add your OPENAI_API_KEY to
.env to switch to GPT-3.5-turbo-based reasoning.
"""

import os
import re
import json
import math
import textwrap
import numpy as np
import pandas as pd
import networkx as nx
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import Any

import config

# ─────────────────────────────────────────────────────────────────────────────
# MESSAGE TYPES
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class AgentMessage:
    sender    : str
    receiver  : str
    content   : Any
    msg_type  : str = "text"          # "text" | "data" | "action" | "result"
    timestamp : str = field(
        default_factory=lambda: datetime.now().isoformat(timespec="seconds")
    )

    def __str__(self):
        return (f"[{self.timestamp}] {self.sender} → {self.receiver} "
                f"({self.msg_type}): {str(self.content)[:120]}")


# ─────────────────────────────────────────────────────────────────────────────
# BASE AGENT
# ─────────────────────────────────────────────────────────────────────────────

class BaseAgent:
    """Abstract base class for all agents."""

    def __init__(self, name: str):
        self.name     = name
        self.memory   : list[AgentMessage] = []
        self.log_file = os.path.join(config.REPORTS_DIR, f"agent_{name.lower().replace(' ', '_')}.log")

    def receive(self, msg: AgentMessage) -> None:
        self.memory.append(msg)
        self._log(f"RECV  {msg}")

    def send(self, receiver: str, content: Any, msg_type: str = "text") -> AgentMessage:
        msg = AgentMessage(sender=self.name, receiver=receiver,
                           content=content, msg_type=msg_type)
        self._log(f"SEND  {msg}")
        return msg

    def think(self, context: dict) -> str:
        raise NotImplementedError

    def act(self, context: dict) -> AgentMessage:
        raise NotImplementedError

    def _log(self, text: str) -> None:
        os.makedirs(os.path.dirname(self.log_file), exist_ok=True)
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(text + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# MOCK LLM  (rule-based, no API key needed)
# ─────────────────────────────────────────────────────────────────────────────

class MockLLM:
    """
    A rule-based LLM mock that returns structured reasoning
    without any API calls. Suitable for demos and offline testing.
    """

    @staticmethod
    def complete(prompt: str) -> str:
        """Parse key values from prompt and return structured analysis."""
        lines = prompt.lower().split("\n")

        # Extract numeric values from the prompt
        def _extract(key: str) -> float:
            for line in lines:
                if key in line:
                    nums = re.findall(r"[-+]?\d*\.?\d+", line)
                    if nums:
                        return float(nums[-1])
            return 0.0

        degree       = _extract("degree")
        pagerank     = _extract("pagerank")
        amount       = _extract("avg_tx_amount") or _extract("amount")
        score        = _extract("anomaly_score") or _extract("score")
        is_fraud_val = _extract("is_fraud") or _extract("actual label")

        # Compose reasoning
        reasons = []
        risk    = "LOW"

        if degree > 15:
            reasons.append(f"unusually high connectivity ({int(degree)} connections)")
            risk = "MEDIUM"
        if pagerank > 0.005:
            reasons.append(f"high PageRank score ({pagerank:.4f}), indicating influence in the network")
            risk = "HIGH"
        if amount > 5000:
            reasons.append(f"large average transaction amount (${amount:,.0f})")
            risk = "HIGH"
        if score > 0.7:
            reasons.append(f"composite anomaly score is very high ({score:.3f})")
            risk = "CRITICAL"

        if not reasons:
            reasons = ["all metrics are within normal ranges"]
            risk    = "LOW"

        reason_text = "; ".join(reasons)
        explanation = (
            f"Risk Level: {risk}\n"
            f"Reasoning: This node was flagged because it exhibits {reason_text}. "
            f"In financial fraud networks, these characteristics are strongly "
            f"associated with money laundering or unauthorized transaction patterns.\n"
            f"Recommendation: {'Immediate investigation required.' if risk in ('HIGH','CRITICAL') else 'Continue monitoring.'}"
        )
        return explanation


# ─────────────────────────────────────────────────────────────────────────────
# REAL LLM  (OpenAI — optional)
# ─────────────────────────────────────────────────────────────────────────────

class RealLLM:
    """Calls OpenAI ChatCompletion. Requires OPENAI_API_KEY in env."""

    def __init__(self):
        try:
            from openai import OpenAI
            self.client = OpenAI(api_key=config.OPENAI_API_KEY)
        except ImportError:
            raise ImportError("Install openai: pip install openai")

    def complete(self, prompt: str) -> str:
        resp = self.client.chat.completions.create(
            model       = config.LLM_MODEL,
            temperature = config.LLM_TEMPERATURE,
            messages    = [
                {"role": "system",
                 "content": (
                     "You are an expert fraud analyst for a financial bank. "
                     "Analyse the given node profile and explain in clear language "
                     "why it may or may not be fraudulent. Be concise (≤150 words)."
                 )},
                {"role": "user", "content": prompt},
            ],
        )
        return resp.choices[0].message.content


def get_llm():
    if config.USE_MOCK_LLM or not config.OPENAI_API_KEY:
        print("[LLM]  Using MockLLM (rule-based, no API key needed)")
        return MockLLM()
    print("[LLM]  Using OpenAI GPT")
    return RealLLM()


# ─────────────────────────────────────────────────────────────────────────────
# AGENT 1 : DATA ANALYZER AGENT
# ─────────────────────────────────────────────────────────────────────────────

class DataAnalyzerAgent(BaseAgent):
    """
    Responsibilities:
    • Understand graph topology
    • Generate per-node statistical profile
    • Identify risk factors from raw features
    """

    def __init__(self):
        super().__init__("DataAnalyzerAgent")

    def think(self, context: dict) -> str:
        df = context["features_df"]
        n_nodes   = len(df)
        n_fraud   = df["is_fraud"].sum()
        avg_deg   = df["degree"].mean()
        max_deg   = df["degree"].max()
        avg_amt   = df["avg_tx_amount"].mean()

        summary = (
            f"GRAPH ANALYSIS SUMMARY\n"
            f"{'='*45}\n"
            f"Total nodes        : {n_nodes}\n"
            f"Known fraud nodes  : {n_fraud} ({n_fraud/n_nodes*100:.1f}%)\n"
            f"Avg degree         : {avg_deg:.2f}\n"
            f"Max degree         : {max_deg}\n"
            f"Avg transaction amt: ${avg_amt:,.0f}\n"
            f"Density            : {context.get('density', 'N/A')}\n"
            f"{'='*45}"
        )
        return summary

    def profile_node(self, node_id: str, df: pd.DataFrame) -> dict:
        """Return a structured profile for a single node."""
        row = df[df["account_id"] == node_id]
        if row.empty:
            return {"error": f"Node {node_id} not found"}
        r = row.iloc[0].to_dict()
        profile = {
            "account_id"             : r["account_id"],
            "account_type"           : r.get("account_type", "unknown"),
            "degree"                 : int(r["degree"]),
            "in_degree"              : int(r["in_degree"]),
            "out_degree"             : int(r["out_degree"]),
            "in_out_ratio"           : float(r["in_out_ratio"]),
            "clustering_coef"        : float(r["clustering_coef"]),
            "betweenness_centrality" : float(r["betweenness_centrality"]),
            "pagerank"               : float(r["pagerank"]),
            "avg_tx_amount"          : float(r["avg_tx_amount"]),
            "max_tx_amount"          : float(r["max_tx_amount"]),
            "total_tx_volume"        : float(r["total_tx_volume"]),
            "anomaly_score"          : float(r.get("anomaly_score", 0)),
            "is_fraud"               : int(r["is_fraud"]),
        }
        return profile

    def act(self, context: dict) -> AgentMessage:
        summary = self.think(context)
        print(f"\n[{self.name}]\n{summary}")
        return self.send("Orchestrator", summary, msg_type="result")


# ─────────────────────────────────────────────────────────────────────────────
# AGENT 2 : ANOMALY DETECTION AGENT
# ─────────────────────────────────────────────────────────────────────────────

class AnomalyDetectionAgent(BaseAgent):
    """
    Responsibilities:
    • Run multiple detection heuristics
    • Compute risk tiers: CRITICAL / HIGH / MEDIUM / LOW
    • Return ranked list of suspicious nodes
    """

    def __init__(self):
        super().__init__("AnomalyDetectionAgent")

    def think(self, context: dict) -> str:
        df     = context["results_df"]
        top_n  = context.get("top_n", 10)

        # Use ensemble prediction + anomaly score
        suspicious = df[df["pred_ensemble"] == 1].sort_values(
            "anomaly_score", ascending=False
        ).head(top_n)

        report_lines = [
            f"TOP {top_n} SUSPICIOUS NODES (Ensemble + Graph Score)",
            "=" * 55,
            f"{'Rank':<5} {'AccountID':<12} {'Score':<8} {'Degree':<8} "
            f"{'PageRank':<10} {'TrueLabel':<10}",
            "-" * 55,
        ]
        for rank, (_, row) in enumerate(suspicious.iterrows(), 1):
            report_lines.append(
                f"{rank:<5} {row['account_id']:<12} "
                f"{row['anomaly_score']:<8.4f} {int(row['degree']):<8} "
                f"{row['pagerank']:<10.6f} "
                f"{'FRAUD' if row['is_fraud'] else 'NORMAL':<10}"
            )

        return "\n".join(report_lines)

    def classify_risk(self, score: float) -> str:
        if score > 0.85:  return "CRITICAL"
        if score > 0.70:  return "HIGH"
        if score > 0.50:  return "MEDIUM"
        return "LOW"

    def act(self, context: dict) -> AgentMessage:
        report = self.think(context)
        print(f"\n[{self.name}]\n{report}")
        df = context["results_df"]
        # Return top flagged nodes as data
        suspicious = df[df["pred_ensemble"] == 1].sort_values(
            "anomaly_score", ascending=False
        )
        payload = {
            "report"     : report,
            "suspicious" : suspicious["account_id"].tolist(),
            "scores"     : suspicious["anomaly_score"].tolist(),
        }
        return self.send("Orchestrator", payload, msg_type="data")


# ─────────────────────────────────────────────────────────────────────────────
# AGENT 3 : EXPLANATION AGENT
# ─────────────────────────────────────────────────────────────────────────────

class ExplanationAgent(BaseAgent):
    """
    Responsibilities:
    • For each flagged node, generate a human-readable explanation
    • Use LLM (mock or real) for natural-language reasoning
    • Produce a PDF-ready audit report
    """

    def __init__(self, llm):
        super().__init__("ExplanationAgent")
        self.llm = llm

    def _build_prompt(self, profile: dict) -> str:
        lines = [
            "Analyse this bank account for potential fraud:",
            "",
            f"Account ID       : {profile['account_id']}",
            f"Account type     : {profile['account_type']}",
            f"Degree           : {profile['degree']}",
            f"In-degree        : {profile['in_degree']}",
            f"Out-degree       : {profile['out_degree']}",
            f"In-out ratio     : {profile['in_out_ratio']:.4f}",
            f"Clustering coef  : {profile['clustering_coef']:.4f}",
            f"Betweenness      : {profile['betweenness_centrality']:.6f}",
            f"PageRank         : {profile['pagerank']:.6f}",
            f"Avg tx amount    : ${profile['avg_tx_amount']:,.2f}",
            f"Max tx amount    : ${profile['max_tx_amount']:,.2f}",
            f"Total volume     : ${profile['total_tx_volume']:,.2f}",
            f"Anomaly score    : {profile['anomaly_score']:.4f}",
            f"Actual label     : {'FRAUD' if profile['is_fraud'] else 'NORMAL'}",
            "",
            "Provide: (1) Risk level, (2) Reasons, (3) Recommendation",
        ]
        return "\n".join(lines)

    def explain(self, profile: dict) -> str:
        prompt = self._build_prompt(profile)
        return self.llm.complete(prompt)

    def act(self, context: dict) -> AgentMessage:
        profiles    = context["profiles"]
        top_n       = min(5, len(profiles))
        explanations = {}

        print(f"\n[{self.name}]  Generating explanations for top-{top_n} nodes …")
        for profile in profiles[:top_n]:
            node_id = profile["account_id"]
            expl    = self.explain(profile)
            explanations[node_id] = expl
            print(f"\n  ── {node_id} ────────────────────────────────")
            print(textwrap.indent(expl, "  "))

        return self.send("Orchestrator", explanations, msg_type="result")


# ─────────────────────────────────────────────────────────────────────────────
# ORCHESTRATOR AGENT
# ─────────────────────────────────────────────────────────────────────────────

class OrchestratorAgent(BaseAgent):
    """
    Coordinates the multi-agent pipeline:
    Step 1 → DataAnalyzerAgent   : understand graph
    Step 2 → AnomalyDetectionAgent: find suspicious nodes
    Step 3 → ExplanationAgent    : explain top suspects
    Step 4 → Compile final report
    """

    def __init__(self):
        super().__init__("OrchestratorAgent")
        self.llm                  = get_llm()
        self.data_analyzer        = DataAnalyzerAgent()
        self.anomaly_detector     = AnomalyDetectionAgent()
        self.explainer            = ExplanationAgent(self.llm)

    def think(self, context: dict) -> str:
        return "Orchestrator planning multi-agent fraud detection pipeline."

    def act(self, context: dict) -> AgentMessage:
        print("\n" + "═" * 65)
        print("   AGENTIC AI FRAUD DETECTION SYSTEM — STARTING")
        print("═" * 65)

        # ─ Step 1: Data Analysis ─────────────────────────────────────────────
        print("\n▶  STEP 1 / 3 — Data Analysis")
        G           = context.get("graph")
        features_df = context["features_df"]
        results_df  = context["results_df"]

        density = nx.density(G) if G else "N/A"
        analysis_ctx = {"features_df": features_df, "density": f"{density:.4f}"}
        msg1 = self.data_analyzer.act(analysis_ctx)
        self.receive(msg1)

        # ─ Step 2: Anomaly Detection ─────────────────────────────────────────
        print("\n▶  STEP 2 / 3 — Anomaly Detection")
        detection_ctx = {"results_df": results_df, "top_n": 15}
        msg2 = self.anomaly_detector.act(detection_ctx)
        self.receive(msg2)
        payload = msg2.content

        # ─ Step 3: Build node profiles for top suspects ───────────────────────
        print("\n▶  STEP 3 / 3 — Explanation Generation")
        top_nodes = payload["suspicious"][:8]
        profiles  = [
            self.data_analyzer.profile_node(nid, results_df)
            for nid in top_nodes
        ]
        explanation_ctx = {"profiles": profiles}
        msg3 = self.explainer.act(explanation_ctx)
        self.receive(msg3)

        # ─ Final Report ────────────────────────────────────────────────────────
        report = self._compile_report(
            analysis_summary = msg1.content,
            detection_report = payload["report"],
            explanations     = msg3.content,
            results_df       = results_df,
        )
        return self.send("User", report, msg_type="result")

    def _compile_report(
        self, analysis_summary, detection_report, explanations, results_df
    ) -> str:
        fraud_found   = int(results_df["pred_ensemble"].sum())
        true_fraud    = int(results_df["is_fraud"].sum())
        detected_true = int(
            (results_df["pred_ensemble"] & results_df["is_fraud"]).sum()
        )

        report = "\n".join([
            "",
            "╔" + "═" * 63 + "╗",
            "║   FRAUD DETECTION FINAL REPORT — AGENTIC AI SYSTEM        ║",
            "╚" + "═" * 63 + "╝",
            "",
            analysis_summary,
            "",
            detection_report,
            "",
            "─" * 65,
            "EXPLANATION HIGHLIGHTS (Top Flagged Nodes)",
            "─" * 65,
        ])

        for node_id, expl in list(explanations.items())[:3]:
            report += f"\n\n  Account: {node_id}\n"
            report += textwrap.indent(expl, "  ")

        report += (
            f"\n\n{'─'*65}\n"
            f"SUMMARY\n"
            f"  Total accounts analysed : {len(results_df)}\n"
            f"  Flagged by system       : {fraud_found}\n"
            f"  Actual fraud accounts   : {true_fraud}\n"
            f"  True positives detected : {detected_true} "
            f"({detected_true/true_fraud*100:.1f}% recall)\n"
            f"{'─'*65}\n"
        )

        # Save report
        report_path = os.path.join(config.REPORTS_DIR, "final_report.txt")
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report)
        print(f"\n[REPORT]  Saved → {report_path}")
        return report


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import pickle

    # Load graph
    graph_path = os.path.join(config.DATA_DIR, "graph.pkl")
    if not os.path.exists(graph_path):
        raise FileNotFoundError("Run part1_dataset.py first.")
    with open(graph_path, "rb") as f:
        G, accounts_df, transactions_df = pickle.load(f)

    # Load features + detection results
    features_path = os.path.join(config.DATA_DIR, "node_features.csv")
    results_path  = os.path.join(config.DATA_DIR, "detection_results.csv")

    if not os.path.exists(results_path):
        raise FileNotFoundError("Run part3_fraud_detection.py first.")

    features_df = pd.read_csv(features_path)
    results_df  = pd.read_csv(results_path)

    # Run the multi-agent system
    orchestrator = OrchestratorAgent()
    context = {
        "graph"       : G,
        "features_df" : features_df,
        "results_df"  : results_df,
    }
    final_msg = orchestrator.act(context)
    print("\n" + "═" * 65)
    print(final_msg.content)
