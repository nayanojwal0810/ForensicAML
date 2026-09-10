"""
src/evaluation/metrics.py
Standardized evaluation metrics for AML transaction surveillance.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score, brier_score_loss


@dataclass(frozen=True)
class EvaluationReport:
    pr_auc: float
    roc_auc: float
    brier_score: float
    recall_at_1pct: float
    recall_at_2pct: float

    def to_dict(self) -> Dict[str, float]:
        return {
            "pr_auc": round(self.pr_auc, 4),
            "roc_auc": round(self.roc_auc, 4),
            "brier_score": round(self.brier_score, 4),
            "recall@1%": round(self.recall_at_1pct, 4),
            "recall@2%": round(self.recall_at_2pct, 4),
        }


class ModelEvaluator:
    """Computes enterprise ranking and calibration metrics under class imbalance."""

    @staticmethod
    def evaluate(y_true: np.ndarray, y_prob: np.ndarray) -> EvaluationReport:
        y_true = np.asarray(y_true, dtype=int)
        y_prob = np.asarray(y_prob, dtype=float)

        # Ranking & calibration metrics
        pr_auc = float(average_precision_score(y_true, y_prob))
        roc_auc = float(roc_auc_score(y_true, y_prob))
        brier = float(brier_score_loss(y_true, y_prob))

        # Fixed investigation capacity metrics (Top K% review queue)
        n_total = len(y_true)
        k_1pct = max(1, int(0.01 * n_total))
        k_2pct = max(1, int(0.02 * n_total))

        # Sort indices by predicted probability descending
        sorted_indices = np.argsort(-y_prob)
        total_positives = np.sum(y_true == 1)

        recall_1pct = (
            float(np.sum(y_true[sorted_indices[:k_1pct]] == 1) / total_positives)
            if total_positives > 0
            else 0.0
        )
        recall_2pct = (
            float(np.sum(y_true[sorted_indices[:k_2pct]] == 1) / total_positives)
            if total_positives > 0
            else 0.0
        )

        return EvaluationReport(
            pr_auc=pr_auc,
            roc_auc=roc_auc,
            brier_score=brier,
            recall_at_1pct=recall_1pct,
            recall_at_2pct=recall_2pct,
        )
