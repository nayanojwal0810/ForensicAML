"""
src/evaluation/graph_validation_gate.py
Focused validation gate for graph topology, temporal integrity, and feature utility.
"""

from __future__ import annotations
import networkx as nx
import numpy as np
import pandas as pd
from scipy.stats import spearmanr


class GraphValidationGate:
    """Executes structural and temporal integrity checks on graph features."""

    def __init__(self, full_nodes: pd.DataFrame, edges: pd.DataFrame, train_m1: pd.DataFrame) -> None:
        self.full_nodes = full_nodes[["txId", "time_step"]].copy()
        self.edges = edges[["source", "target"]].copy()
        self.train_m1 = train_m1.copy()
        self.node_time_map = dict(zip(self.full_nodes["txId"], self.full_nodes["time_step"]))

    def audit_cycles_and_sccs(self) -> pd.DataFrame:
        """Gate 1: Checks for cycles and SCCs > 1 across time steps and full graph."""
        records = []
        
        # 1. Full Graph Check
        G_full = nx.DiGraph()
        G_full.add_edges_from(zip(self.edges["source"], self.edges["target"]))
        is_full_dag = nx.is_directed_acyclic_graph(G_full)
        full_sccs_gt1 = sum(1 for c in nx.strongly_connected_components(G_full) if len(c) > 1)

        print(f"[AUDIT] Full Graph (All 49 Steps):")
        print(f"  - Total Nodes: {G_full.number_of_nodes():,}")
        print(f"  - Total Edges: {G_full.number_of_edges():,}")
        print(f"  - Is Strictly Directed Acyclic Graph (DAG): {is_full_dag}")
        print(f"  - Strongly Connected Components > 1: {full_sccs_gt1}")

        # 2. Per-Time-Step Audit
        edge_src_time = self.edges["source"].map(self.node_time_map)
        edge_tgt_time = self.edges["target"].map(self.node_time_map)
        
        for t in range(1, 50):
            nodes_t = set(self.full_nodes[self.full_nodes["time_step"] == t]["txId"])
            mask_t = (edge_src_time == t) & (edge_tgt_time == t)
            sub_edges = self.edges[mask_t]

            G_t = nx.DiGraph()
            G_t.add_nodes_from(nodes_t)
            G_t.add_edges_from(zip(sub_edges["source"], sub_edges["target"]))

            sccs = list(nx.strongly_connected_components(G_t))
            sccs_gt1 = [c for c in sccs if len(c) > 1]
            max_scc = max((len(c) for c in sccs), default=1)

            records.append({
                "time_step": t,
                "nodes": len(nodes_t),
                "edges": len(sub_edges),
                "scc_gt1_count": len(sccs_gt1),
                "max_scc_size": max_scc,
            })

        return pd.DataFrame(records)

    def audit_temporal_edges(self) -> dict:
        """Gate 2: Verifies whether edges cross time boundaries."""
        src_time = self.edges["source"].map(self.node_time_map)
        tgt_time = self.edges["target"].map(self.node_time_map)

        same_time = int((src_time == tgt_time).sum())
        forward_time = int((src_time < tgt_time).sum())
        backward_time = int((src_time > tgt_time).sum())
        unmapped = int((src_time.isna() | tgt_time.isna()).sum())

        return {
            "total_edges": len(self.edges),
            "edges_same_time_step": same_time,
            "edges_forward_in_time": forward_time,
            "edges_backward_in_time": backward_time,
            "edges_unmapped": unmapped,
            "point_in_time_safe": bool(backward_time == 0 and unmapped == 0),
        }

    def audit_train_feature_usefulness(self) -> pd.DataFrame:
        """Gate 3: Evaluates variance, missingness, unique values, and target correlation."""
        graph_cols = ["in_degree", "out_degree", "pagerank", "deg_imbalance", "pass_through_ratio", "cycle_flag", "scc_size"]
        y_train = self.train_m1["label"].to_numpy(dtype=int)

        summary = []
        for col in graph_cols:
            if col not in self.train_m1.columns:
                continue
            vals = self.train_m1[col].to_numpy(dtype=float)
            var = float(np.var(vals))
            null_count = int(self.train_m1[col].isna().sum())
            unique_vals = int(len(np.unique(vals)))
            
            # Spearman rank correlation with label
            if unique_vals > 1:
                corr, p_val = spearmanr(vals, y_train)
            else:
                corr, p_val = 0.0, 1.0

            status = "HEALTHY"
            if unique_vals <= 1 or var == 0:
                status = "ZERO_VARIANCE_DROP"
            elif unique_vals < 5:
                status = "NEAR_CONSTANT"

            summary.append({
                "feature": col,
                "variance": round(var, 6),
                "null_count": null_count,
                "unique_values": unique_vals,
                "spearman_corr": round(float(corr), 4),
                "p_value": float(f"{p_val:.2e}"),
                "status": status,
            })

        return pd.DataFrame(summary)
