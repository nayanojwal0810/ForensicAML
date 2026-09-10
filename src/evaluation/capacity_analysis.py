"""
src/evaluation/capacity_analysis.py
Fixed-capacity queue analysis for AML operations.
"""

from __future__ import annotations
from typing import List, Union
import numpy as np
import pandas as pd
from src.evaluation.cost_optimizer import CostOptimizer, ThresholdResult


class CapacityAnalyzer:
    """Evaluates metrics across operational review capacity constraints."""

    def __init__(self, optimizer: CostOptimizer) -> None:
        self.optimizer = optimizer

    def analyze_capacities(
        self, 
        y_true: np.ndarray, 
        y_prob: np.ndarray, 
        capacities: List[float], 
        c_inv: float, 
        c_loss: Union[float, np.ndarray]
    ) -> pd.DataFrame:
        """Evaluates model performance given fixed top-K% capacity limits."""
        y_true = np.asarray(y_true, dtype=int)
        y_prob = np.asarray(y_prob, dtype=float)
        
        n_total = len(y_true)
        sorted_indices = np.argsort(-y_prob)
        
        results = []
        for cap_pct in capacities:
            k = max(1, int(n_total * cap_pct))
            
            # The implied threshold is the probability of the k-th highest ranked transaction
            implied_threshold = float(y_prob[sorted_indices[k - 1]])
            
            res: ThresholdResult = self.optimizer.evaluate_threshold(
                y_true, y_prob, implied_threshold, c_inv, c_loss, f"Top {cap_pct*100}% Review"
            )
            
            results.append({
                "Capacity Policy": res.policy_name,
                "Implied Threshold": round(res.threshold, 4),
                "Cases Reviewed": res.investigations,
                "Illicit Captured (TP)": res.tp,
                "Recall (%)": round(res.recall * 100, 2),
                "Precision (%)": round(res.precision * 100, 2),
                "Expected Cost ($)": round(res.total_cost, 2),
                "Missed Exposure ($)": round(res.missed_loss, 2)
            })
            
        return pd.DataFrame(results)
