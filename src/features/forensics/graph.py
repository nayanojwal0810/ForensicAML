"""
src/features/forensics/graph.py
Point-in-time graph topology and forensic flow feature extraction.
"""

from __future__ import annotations
import networkx as nx
import numpy as np
import pandas as pd


class TemporalGraphFeatureExtractor:
    """Extracts scalable graph-topology forensic features on temporal prefix slices."""

    def __init__(self, full_nodes: pd.DataFrame, edges: pd.DataFrame) -> None:
        self.full_nodes = full_nodes[["txId", "time_step"]].copy()
        self.edges = edges[["source", "target"]].copy()

    def extract_features(self) -> pd.DataFrame:
        """Computes topological indicators strictly using known temporal edges."""
        # Map node to time_step for point-in-time filtering
        node_time_map = dict(zip(self.full_nodes["txId"], self.full_nodes["time_step"]))

        # Filter edges: only retain edges where both nodes exist within the known universe
        # Elliptic edges occur primarily within the same time step or immediately preceding
        edge_time = self.edges["source"].map(node_time_map)
        valid_edges = self.edges.dropna().copy()
        valid_edges["time_step"] = edge_time

        feature_records: list[pd.DataFrame] = []

        print("[INFO] Computing temporal graph topology across 49 time steps...")
        # Process each time step's graph incrementally (strict temporal isolation)
        for t in range(1, 50):
            # Nodes present up to time t
            nodes_t = set(self.full_nodes[self.full_nodes["time_step"] == t]["txId"])
            if not nodes_t:
                continue

            # Active edges for the graph up to time t (or local slice)
            sub_edges = valid_edges[valid_edges["time_step"] == t]

            # Construct directed graph
            G = nx.DiGraph()
            G.add_nodes_from(nodes_t)
            edge_tuples = [
                (src, tgt) for src, tgt in zip(sub_edges["source"], sub_edges["target"])
                if src in nodes_t and tgt in nodes_t
            ]
            G.add_edges_from(edge_tuples)

            # 1. Degree metrics
            in_deg = dict(G.in_degree())
            out_deg = dict(G.out_degree())

            # 2. Scalable PageRank
            try:
                pr = nx.pagerank(G, alpha=0.85, max_iter=100, tol=1e-4)
            except Exception:
                pr = {n: 0.0 for n in nodes_t}

            # 3. Strongly Connected Components (cycle proxies)
            sccs = list(nx.strongly_connected_components(G))
            scc_map = {}
            for comp in sccs:
                comp_size = len(comp)
                for node in comp:
                    scc_map[node] = comp_size

            # Synthesize per-node metrics
            batch_df = pd.DataFrame({"txId": list(nodes_t)})
            batch_df["in_degree"] = batch_df["txId"].map(in_deg).fillna(0).astype(int)
            batch_df["out_degree"] = batch_df["txId"].map(out_deg).fillna(0).astype(int)
            batch_df["pagerank"] = batch_df["txId"].map(pr).fillna(0.0).astype(float)
            batch_df["scc_size"] = batch_df["txId"].map(scc_map).fillna(1).astype(int)

            # Degree imbalance: |out - in| / (1 + out + in)
            deg_diff = (batch_df["out_degree"] - batch_df["in_degree"]).abs()
            deg_sum = batch_df["out_degree"] + batch_df["in_degree"] + 1
            batch_df["deg_imbalance"] = deg_diff / deg_sum

            # Pass-through conduit score: min(in, out) / (max(in, out) + 1)
            min_deg = np.minimum(batch_df["in_degree"], batch_df["out_degree"])
            max_deg = np.maximum(batch_df["in_degree"], batch_df["out_degree"]) + 1
            batch_df["pass_through_ratio"] = min_deg / max_deg

            # Cycle indicator: 1 if participating in a multi-node cycle (SCC > 1)
            batch_df["cycle_flag"] = (batch_df["scc_size"] > 1).astype(int)

            feature_records.append(batch_df)

        full_graph_features = pd.concat(feature_records, ignore_index=True)
        print(f"[INFO] Graph forensic features generated for {len(full_graph_features):,} nodes.")
        return full_graph_features
