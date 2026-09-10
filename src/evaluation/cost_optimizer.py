"""
src/evaluation/cost_optimizer.py
Business cost matrix and capacity-constrained threshold optimization.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Union
import numpy as np


@dataclass(frozen=True)
class ThresholdResult:
    policy_name: str
    threshold: float
    total_cost: float
    fp: int
    fn: int
    tp: int
    tn: int
    investigations: int
    investigation_cost: float
    missed_loss: float
    recall: float
    precision: float


class CostOptimizer:
    """Evaluates and optimizes classification thresholds based on business cost functions."""

    def evaluate_threshold(
        self, 
        y_true: np.ndarray, 
        y_prob: np.ndarray, 
        threshold: float, 
        c_inv: float, 
        c_loss: Union[float, np.ndarray], 
        policy_name: str = "Custom"
    ) -> ThresholdResult:
        """Evaluates financial impact of a specific probability threshold."""
        pred = (y_prob >= threshold).astype(int)
        
        tp = int(np.sum((pred == 1) & (y_true == 1)))
        fp = int(np.sum((pred == 1) & (y_true == 0)))
        tn = int(np.sum((pred == 0) & (y_true == 0)))
        fn_mask = (pred == 0) & (y_true == 1)
        fn = int(np.sum(fn_mask))
        
        investigations = tp + fp
        inv_cost = fp * c_inv
        
        if isinstance(c_loss, np.ndarray):
            miss_cost = float(np.sum(c_loss[fn_mask]))
        else:
            miss_cost = fn * c_loss
            
        total_cost = inv_cost + miss_cost
        
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0

        return ThresholdResult(
            policy_name=policy_name,
            threshold=threshold,
            total_cost=total_cost,
            fp=fp, fn=fn, tp=tp, tn=tn,
            investigations=investigations,
            investigation_cost=inv_cost,
            missed_loss=miss_cost,
            recall=recall,
            precision=precision,
        )

    def optimize_unconstrained(
        self, 
        y_true: np.ndarray, 
        y_prob: np.ndarray, 
        c_inv: float, 
        c_loss: Union[float, np.ndarray]
    ) -> ThresholdResult:
        """Finds the threshold minimizing absolute expected cost."""
        thresholds = np.linspace(0.001, 0.999, 999)
        best_res: Optional[ThresholdResult] = None
        
        for t in thresholds:
            res = self.evaluate_threshold(y_true, y_prob, t, c_inv, c_loss, "Optimal Unconstrained")
            if best_res is None or res.total_cost < best_res.total_cost:
                best_res = res
                
        if best_res is None:
            raise ValueError("Optimization failed to find a valid threshold.")
        return best_res

    def optimize_capacity_constrained(
        self, 
        y_true: np.ndarray, 
        y_prob: np.ndarray, 
        c_inv: float, 
        c_loss: Union[float, np.ndarray], 
        max_capacity_pct: float
    ) -> ThresholdResult:
        """Finds the lowest cost threshold respecting a maximum investigation queue size."""
        thresholds = np.linspace(0.001, 0.999, 999)
        max_cases = int(len(y_true) * max_capacity_pct)
        best_res: Optional[ThresholdResult] = None
        
        for t in thresholds:
            res = self.evaluate_threshold(
                y_true, y_prob, t, c_inv, c_loss, f"Capacity Constrained ({max_capacity_pct*100}%)"
            )
            if res.investigations <= max_cases:
                if best_res is None or res.total_cost < best_res.total_cost:
                    best_res = res
                    
        if best_res is None:
            return self.evaluate_threshold(y_true, y_prob, 0.999, c_inv, c_loss, "Failsafe (Max Threshold)")
        return best_res
