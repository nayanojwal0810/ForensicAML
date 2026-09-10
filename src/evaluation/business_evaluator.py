"""
src/evaluation/business_evaluator.py
Operational business evaluation, Top-K deterministic queueing, and financial baselining.
"""

from __future__ import annotations
from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd


class BusinessEvaluator:
    """Evaluates financial impact of classification models at fixed operational capacities."""

    def __init__(self, c_inv: float = 50.0, c_loss: float = 2000.0) -> None:
        self.c_inv = c_inv
        self.c_loss = c_loss

    def get_top_k_indices(self, probs: np.ndarray, tx_ids: np.ndarray, k: int) -> np.ndarray:
        """Returns indices of top K predictions, deterministically tie-broken by txId."""
        # lexsort sorts by the last key first: tx_ids (ascending), then -probs (descending)
        return np.lexsort((tx_ids, -probs))[:k]

    def audit_rank_preservation(
        self, raw_probs: np.ndarray, cal_probs: np.ndarray, tx_ids: np.ndarray, k: int
    ) -> Dict[str, Any]:
        """Verifies if probability calibration strictly preserves rank ordering."""
        raw_idx = self.get_top_k_indices(raw_probs, tx_ids, k)
        cal_idx = self.get_top_k_indices(cal_probs, tx_ids, k)
        
        overlap = len(set(raw_idx).intersection(set(cal_idx)))
        return {
            "k_expected": k,
            "raw_k_actual": len(raw_idx),
            "cal_k_actual": len(cal_idx),
            "exact_overlap": overlap,
            "preserves_ranking": overlap == k
        }

    def evaluate_fixed_capacity(
        self, y_true: np.ndarray, probs: np.ndarray, tx_ids: np.ndarray, capacity_pct: float
    ) -> Dict[str, Any]:
        """Calculates TP, FP, Precision, Recall, and expected costs for a Top-K% policy."""
        n_total = len(y_true)
        k = int(np.ceil(capacity_pct * n_total))
        total_illicit = int(np.sum(y_true == 1))
        
        top_k_idx = self.get_top_k_indices(probs, tx_ids, k)
        implied_threshold = float(probs[top_k_idx[-1]])
        
        tp = int(np.sum(y_true[top_k_idx] == 1))
        fp = k - tp
        fn = total_illicit - tp
        tn = (n_total - total_illicit) - fp
        
        prec = tp / k if k > 0 else 0.0
        rec = tp / total_illicit if total_illicit > 0 else 0.0
        
        fp_cost = fp * self.c_inv
        fn_cost = fn * self.c_loss
        total_cost = fp_cost + fn_cost
        
        return {
            "capacity_pct": capacity_pct,
            "k_volume": k,
            "implied_threshold": implied_threshold,
            "tp": tp, "fp": fp, "tn": tn, "fn": fn,
            "precision": prec,
            "recall": rec,
            "fp_cost": fp_cost,
            "fn_cost": fn_cost,
            "total_expected_cost": total_cost
        }

    def evaluate_random_baseline(
        self, y_true: np.ndarray, capacity_pct: float
    ) -> Dict[str, Any]:
        """Calculates mathematical expectation of costs for a random selection baseline."""
        n_total = len(y_true)
        k = int(np.ceil(capacity_pct * n_total))
        total_illicit = int(np.sum(y_true == 1))
        prevalence = total_illicit / n_total if n_total > 0 else 0.0
        
        tp_rnd = k * prevalence
        fp_rnd = k - tp_rnd
        fn_rnd = total_illicit - tp_rnd
        
        rec_rnd = tp_rnd / total_illicit if total_illicit > 0 else 0.0
        
        fp_cost = fp_rnd * self.c_inv
        fn_cost = fn_rnd * self.c_loss
        total_cost = fp_cost + fn_cost
        
        return {
            "policy": f"Random {capacity_pct*100}%",
            "tp": tp_rnd, "fp": fp_rnd, "fn": fn_rnd,
            "precision": prevalence,
            "recall": rec_rnd,
            "total_expected_cost": total_cost
        }
