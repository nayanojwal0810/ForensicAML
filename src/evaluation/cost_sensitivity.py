"""
src/evaluation/cost_sensitivity.py
Stress-testing optimal thresholds under varying business cost assumptions.
"""

from __future__ import annotations
from typing import List, Tuple
import numpy as np
import pandas as pd
from src.evaluation.cost_optimizer import CostOptimizer


class CostSensitivityAnalyzer:
    """Simulates financial outcomes across multiple cost assumption scenarios."""

    def __init__(self, optimizer: CostOptimizer) -> None:
        self.optimizer = optimizer

    def run_scenarios(
        self, 
        y_true: np.ndarray, 
        y_prob: np.ndarray, 
        scenarios: List[Tuple[float, float]]
    ) -> pd.DataFrame:
        """Executes unconstrained optimization for different (C_inv, C_loss) pairs."""
        results = []
        for c_inv, c_loss in scenarios:
            res = self.optimizer.optimize_unconstrained(y_true, y_prob, c_inv, c_loss)
            
            results.append({
                "C_inv ($)": c_inv,
                "C_loss ($)": c_loss,
                "Optimal Threshold": round(res.threshold, 4),
                "Investigations": res.investigations,
                "Recall (%)": round(res.recall * 100, 2),
                "Precision (%)": round(res.precision * 100, 2),
                "Expected Cost ($)": round(res.total_cost, 2)
            })
            
        return pd.DataFrame(results)
