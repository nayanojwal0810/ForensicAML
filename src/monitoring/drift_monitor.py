"""
src/monitoring/drift_monitor.py
Monitors data drift (PSI), prediction drift, and temporal performance degradation.
"""

from __future__ import annotations
from typing import Dict, List, Any
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from src.evaluation.metrics import ModelEvaluator


class DriftMonitor:
    """Tracks feature and prediction stability across reference and target distributions."""

    @staticmethod
    def compute_psi(ref: np.ndarray, tgt: np.ndarray, bins: int = 10) -> float:
        """Calculates Population Stability Index using reference quantiles."""
        ref = ref[~np.isnan(ref)]
        tgt = tgt[~np.isnan(tgt)]
        
        if len(ref) == 0 or len(tgt) == 0:
            return 0.0

        percentiles = np.linspace(0, 100, bins + 1)
        bin_edges = np.percentile(ref, percentiles)
        bin_edges[0] = -np.inf
        bin_edges[-1] = np.inf
        bin_edges = np.unique(bin_edges)

        if len(bin_edges) < 2:
            return 0.0

        p, _ = np.histogram(ref, bins=bin_edges)
        q, _ = np.histogram(tgt, bins=bin_edges)
        
        p_pct = (p + 1e-4) / (sum(p) + 1e-3)
        q_pct = (q + 1e-4) / (sum(q) + 1e-3)
        
        return float(np.sum((q_pct - p_pct) * np.log(q_pct / p_pct)))

    def monitor_features(self, ref_df: pd.DataFrame, tgt_df: pd.DataFrame, features: List[str]) -> pd.DataFrame:
        """Computes PSI for a specific list of features."""
        records = []
        for f in features:
            psi = self.compute_psi(ref_df[f].to_numpy(), tgt_df[f].to_numpy())
            records.append({
                "feature": f,
                "psi": round(psi, 4),
                "status": "High Drift" if psi > 0.25 else ("Moderate" if psi > 0.1 else "Stable")
            })
        return pd.DataFrame(records)

    def monitor_predictions(self, ref_probs: np.ndarray, tgt_probs: np.ndarray) -> Dict[str, Any]:
        """Monitors predicted probability distribution shift."""
        ks_stat, p_val = ks_2samp(ref_probs, tgt_probs)
        psi = self.compute_psi(ref_probs, tgt_probs, bins=10)
        
        return {
            "ref_mean": round(float(np.mean(ref_probs)), 4),
            "tgt_mean": round(float(np.mean(tgt_probs)), 4),
            "ref_p95": round(float(np.percentile(ref_probs, 95)), 4),
            "tgt_p95": round(float(np.percentile(tgt_probs, 95)), 4),
            "prediction_psi": round(psi, 4),
            "ks_statistic": round(float(ks_stat), 4),
            "drift_detected": bool(p_val < 0.01 or psi > 0.25)
        }

    def monitor_performance(self, y_true: np.ndarray, y_prob: np.ndarray, capacity_pct: float = 0.05) -> Dict[str, Any]:
        """Monitors ranking and calibration quality when ground truth arrives."""
        eval_rep = ModelEvaluator.evaluate(y_true, y_prob)
        
        n_total = len(y_true)
        k = int(np.ceil(capacity_pct * n_total))
        total_illicit = int(np.sum(y_true == 1))
        
        top_k_idx = np.argsort(-y_prob)[:k]
        tp = int(np.sum(y_true[top_k_idx] == 1))
        recall_at_k = tp / total_illicit if total_illicit > 0 else 0.0

        return {
            "pr_auc": round(eval_rep.pr_auc, 4),
            "brier_score": round(eval_rep.brier_score, 4),
            "recall_at_5pct": round(recall_at_k, 4)
        }
