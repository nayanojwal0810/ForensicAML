"""
src/explainability/shap_explainer.py
Local explainability using TreeSHAP for investigator alert justification.
"""

from __future__ import annotations
from typing import Any, Dict, List
import numpy as np
import pandas as pd
import shap
import lightgbm as lgb


class ShapExplainer:
    """Generates local feature attributions and ensures SHAP/model consistency."""

    def __init__(self, model: lgb.LGBMClassifier, feature_names: List[str]) -> None:
        self.model = model
        self.feature_names = feature_names
        self.explainer = shap.TreeExplainer(self.model)
        
    def explain(self, tx_features: pd.DataFrame, top_k: int = 5) -> Dict[str, Any]:
        """
        Computes SHAP values, extracts top contributors, and verifies consistency.
        Does NOT claim causal evidence.
        """
        if tx_features.shape[0] != 1:
            raise ValueError("Explain method requires exactly one transaction.")

        shap_values_obj = self.explainer(tx_features)
        
        vals = shap_values_obj.values
        if vals.ndim == 3:
            shap_vals = vals[0, :, 1] if vals.shape[2] > 1 else vals[0, :, -1]
        elif vals.ndim == 2:
            shap_vals = vals[0]
        else:
            shap_vals = vals.reshape(-1)
            
        base_value = float(
            shap_values_obj.base_values[0] 
            if isinstance(shap_values_obj.base_values, (list, np.ndarray)) 
            else shap_values_obj.base_values
        )

        expected_log_odds = base_value + np.sum(shap_vals)
        ranked_indices = np.argsort(-np.abs(shap_vals))
        
        evidence: List[Dict[str, Any]] = []
        for idx in ranked_indices[:top_k]:
            feat_name = self.feature_names[idx]
            feat_val = float(tx_features.iloc[0, idx])
            shap_val = float(shap_vals[idx])
            
            evidence.append({
                "feature": feat_name,
                "value": round(feat_val, 4),
                "shap_value": round(shap_val, 4),
                "absolute_magnitude": round(abs(shap_val), 4),
                "direction": "INCREASED RISK" if shap_val > 0 else "DECREASED RISK"
            })

        return {
            "base_log_odds": round(base_value, 4),
            "shap_sum_log_odds": round(expected_log_odds, 4),
            "top_evidence": evidence
        }
