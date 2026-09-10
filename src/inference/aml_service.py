"""
src/inference/aml_service.py
Production inference service integrating input validation, model, calibrator, and SHAP.
"""

from __future__ import annotations
from typing import Any, Dict, List
import numpy as np
import pandas as pd
from scipy.special import logit
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression

from src.explainability.shap_explainer import ShapExplainer
from src.config import MODEL_VERSION, CALIBRATOR_VERSION, SCHEMA_VERSION

class AmlInferenceService:
    """End-to-end inference service ensuring schema integrity and probability calibration."""

    def __init__(
        self, 
        model: lgb.LGBMClassifier, 
        calibrator: LogisticRegression, 
        feature_schema: List[str], 
        investigation_threshold: float
    ) -> None:
        self.model = model
        self.calibrator = calibrator
        self.schema = feature_schema
        self.threshold = investigation_threshold
        self.explainer = ShapExplainer(model, feature_schema)
        
        self.model_version = MODEL_VERSION
        self.calibrator_version = CALIBRATOR_VERSION
        self.schema_version = SCHEMA_VERSION

    def validate_input(self, x: pd.DataFrame) -> pd.DataFrame:
        """Strictly enforces feature schema, ordering, and data validity."""
        missing = [c for c in self.schema if c not in x.columns]
        if missing:
            raise ValueError(f"Missing {len(missing)} required features.")
        
        x = x[self.schema].copy()
        
        if x.isna().any().any():
            raise ValueError("Input contains NaN values.")
        if np.isinf(x.to_numpy()).any():
            raise ValueError("Input contains Infinity values.")
            
        return x

    def predict(self, x: pd.DataFrame, explain: bool = True) -> Dict[str, Any]:
        """Predicts risk for a single transaction and optionally generates SHAP explanation."""
        x_valid = self.validate_input(x)

        raw_prob = self.model.predict_proba(x_valid)[:, 1]
        clipped = np.clip(raw_prob, 1e-6, 1.0 - 1e-6)
        cal_prob = float(self.calibrator.predict_proba(logit(clipped).reshape(-1, 1))[:, 1][0])
        
        decision = "INVESTIGATE" if cal_prob >= self.threshold else "PASS"

        response = {
            "calibrated_probability": round(cal_prob, 4),
            "decision": decision,
            "policy": f"Fixed 5% Capacity (Threshold: {self.threshold:.4f})",
            "model_version": self.model_version,
            "calibrator_version": self.calibrator_version,
            "schema_version": self.schema_version,
        }

        if explain:
            response["explanation"] = self.explainer.explain(x_valid, top_k=5)

        return response

    def predict_batch(self, X: pd.DataFrame) -> np.ndarray:
        """Efficient batch prediction without explanations."""
        X_valid = self.validate_input(X)
        raw_probs = self.model.predict_proba(X_valid)[:, 1]
        clipped = np.clip(raw_probs, 1e-6, 1.0 - 1e-6)
        cal_probs = self.calibrator.predict_proba(logit(clipped).reshape(-1, 1))[:, 1]
        return cal_probs

