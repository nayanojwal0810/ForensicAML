"""
src/models/calibration.py
Out-of-sample probability calibration and reliability evaluation for AML surveillance.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from scipy.special import logit
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


@dataclass(frozen=True)
class CalibrationMetrics:
    brier_score: float
    ece: float
    pr_auc: float
    roc_auc: float
    slope: float
    intercept: float

    def to_dict(self) -> Dict[str, float]:
        return {
            "brier_score": round(self.brier_score, 4),
            "ece": round(self.ece, 4),
            "pr_auc": round(self.pr_auc, 4),
            "roc_auc": round(self.roc_auc, 4),
            "cal_slope": round(self.slope, 4),
            "cal_intercept": round(self.intercept, 4),
        }


class ProbabilityCalibrator:
    """Manages Platt (Sigmoid) and Isotonic calibration pipelines."""

    def __init__(self, n_bins: int = 10) -> None:
        self.n_bins = n_bins
        self.platt_model: LogisticRegression | None = None
        self.isotonic_model: IsotonicRegression | None = None

    @staticmethod
    def compute_ece(
        y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10
    ) -> Tuple[float, pd.DataFrame]:
        """Calculates Expected Calibration Error and returns bin reliability data."""
        y_true = np.asarray(y_true, dtype=int)
        y_prob = np.asarray(y_prob, dtype=float)

        bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
        bin_indices = np.digitize(y_prob, bin_edges) - 1
        bin_indices = np.clip(bin_indices, 0, n_bins - 1)

        ece = 0.0
        n_total = len(y_true)
        records = []

        for b in range(n_bins):
            mask = bin_indices == b
            bin_size = int(np.sum(mask))

            if bin_size > 0:
                conf = float(np.mean(y_prob[mask]))
                acc = float(np.mean(y_true[mask]))
                weight = bin_size / n_total
                ece += weight * abs(acc - conf)

                records.append({
                    "bin": b,
                    "bin_range": f"[{bin_edges[b]:.1f}, {bin_edges[b+1]:.1f})",
                    "count": bin_size,
                    "mean_confidence": round(conf, 4),
                    "empirical_accuracy": round(acc, 4),
                    "abs_error": round(abs(acc - conf), 4),
                })

        return float(ece), pd.DataFrame(records)

    @staticmethod
    def compute_calibration_curve_stats(
        y_true: np.ndarray, y_prob: np.ndarray
    ) -> Tuple[float, float]:
        """Calculates calibration slope and intercept via logistic regression on log-odds."""
        y_true = np.asarray(y_true, dtype=int)
        eps = 1e-6
        clipped_probs = np.clip(y_prob, eps, 1.0 - eps)
        log_odds = logit(clipped_probs).reshape(-1, 1)

        lr = LogisticRegression(solver="lbfgs", max_iter=1000)
        lr.fit(log_odds, y_true)
        slope = float(lr.coef_[0, 0])
        intercept = float(lr.intercept_[0])
        return slope, intercept

    def evaluate_probs(
        self, y_true: np.ndarray, y_prob: np.ndarray
    ) -> CalibrationMetrics:
        """Computes comprehensive ranking, reliability, and calibration statistics."""
        brier = float(brier_score_loss(y_true, y_prob))
        ece, _ = self.compute_ece(y_true, y_prob, self.n_bins)
        pr_auc = float(average_precision_score(y_true, y_prob))
        roc_auc = float(roc_auc_score(y_true, y_prob))
        slope, intercept = self.compute_calibration_curve_stats(y_true, y_prob)

        return CalibrationMetrics(
            brier_score=brier,
            ece=ece,
            pr_auc=pr_auc,
            roc_auc=roc_auc,
            slope=slope,
            intercept=intercept,
        )

    def fit(self, y_val: np.ndarray, raw_val_probs: np.ndarray) -> None:
        """Fits Platt and Isotonic calibrators strictly on validation predictions."""
        y_val = np.asarray(y_val, dtype=int)
        raw_val_probs = np.asarray(raw_val_probs, dtype=float)

        # 1. Platt Sigmoid Calibration (Logistic on raw log-odds)
        eps = 1e-6
        clipped_val = np.clip(raw_val_probs, eps, 1.0 - eps)
        val_log_odds = logit(clipped_val).reshape(-1, 1)

        self.platt_model = LogisticRegression(solver="lbfgs", max_iter=1000)
        self.platt_model.fit(val_log_odds, y_val)

        # 2. Isotonic Calibration
        self.isotonic_model = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        self.isotonic_model.fit(raw_val_probs, y_val)

    def predict_platt(self, raw_probs: np.ndarray) -> np.ndarray:
        if self.platt_model is None:
            raise RuntimeError("Platt model not fitted.")
        eps = 1e-6
        clipped = np.clip(raw_probs, eps, 1.0 - eps)
        log_odds = logit(clipped).reshape(-1, 1)
        return self.platt_model.predict_proba(log_odds)[:, 1]

    def predict_isotonic(self, raw_probs: np.ndarray) -> np.ndarray:
        if self.isotonic_model is None:
            raise RuntimeError("Isotonic model not fitted.")
        return self.isotonic_model.predict(raw_probs)
