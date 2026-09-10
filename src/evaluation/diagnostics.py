"""
src/evaluation/diagnostics.py
Comprehensive diagnostic suite to investigate temporal performance collapse.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.metrics import average_precision_score, roc_auc_score


class TemporalDiagnosticSuite:
    """Investigates distribution shifts, concept drift, and leakage between temporal splits."""

    def __init__(self, model, train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
        self.model = model
        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df
        
        self.ignore_cols = ["txId", "time_step", "label"]
        self.feature_cols = [c for c in train_df.columns if c not in self.ignore_cols]

    def check_prevalence_by_step(self) -> pd.DataFrame:
        """1. Evaluates class prevalence step-by-step."""
        combined = pd.concat([self.val_df, self.test_df])
        records = []
        for step in sorted(combined["time_step"].unique()):
            sub = combined[combined["time_step"] == step]
            pos = int((sub["label"] == 1).sum())
            total = len(sub)
            records.append({
                "time_step": step,
                "split": "Val" if step <= 41 else "Test",
                "total_rows": total,
                "illicit_count": pos,
                "prevalence_%": round((pos / total) * 100, 2) if total > 0 else 0.0
            })
        return pd.DataFrame(records)

    def check_time_step_performance(self) -> pd.DataFrame:
        """2. Measures step-by-step PR-AUC and ROC-AUC."""
        combined = pd.concat([self.val_df, self.test_df])
        records = []
        for step in sorted(combined["time_step"].unique()):
            sub = combined[combined["time_step"] == step]
            y_true = sub["label"].to_numpy(dtype=int)
            if len(np.unique(y_true)) < 2:
                continue
            X = sub[self.feature_cols].to_numpy(dtype=np.float32)
            y_prob = self.model.predict_proba(X)[:, 1]

            pr_auc = average_precision_score(y_true, y_prob)
            roc_auc = roc_auc_score(y_true, y_prob)
            records.append({
                "time_step": step,
                "split": "Val" if step <= 41 else "Test",
                "pr_auc": round(pr_auc, 4),
                "roc_auc": round(roc_auc, 4),
            })
        return pd.DataFrame(records)

    def check_prediction_drift(self, val_probs: np.ndarray, test_probs: np.ndarray) -> dict:
        """3. Tests for prediction distribution shift."""
        ks_stat, p_val = ks_2samp(val_probs, test_probs)
        return {
            "val_mean_prob": round(float(np.mean(val_probs)), 4),
            "test_mean_prob": round(float(np.mean(test_probs)), 4),
            "val_p95_prob": round(float(np.percentile(val_probs, 95)), 4),
            "test_p95_prob": round(float(np.percentile(test_probs, 95)), 4),
            "ks_statistic": round(float(ks_stat), 4),
            "ks_p_value": float(p_val),
            "drift_detected": bool(p_val < 0.01),
        }

    def check_feature_psi(self, top_n: int = 10) -> pd.DataFrame:
        """4. Computes Population Stability Index (PSI) on top predictive features."""
        importances = self.model.feature_importances_
        top_indices = np.argsort(-importances)[:top_n]
        top_features = [self.feature_cols[i] for i in top_indices]

        psi_results = []
        for col in top_features:
            val_vals = self.val_df[col].dropna().to_numpy()
            test_vals = self.test_df[col].dropna().to_numpy()

            # 10 quantile bins based on validation
            quantiles = np.linspace(0, 100, 11)
            bins = np.percentile(val_vals, quantiles)
            bins[0] = -np.inf
            bins[-1] = np.inf
            bins = np.unique(bins)

            val_counts, _ = np.histogram(val_vals, bins=bins)
            test_counts, _ = np.histogram(test_vals, bins=bins)

            # Proportions with Laplace smoothing
            p = (val_counts + 1e-4) / (len(val_vals) + 1e-3)
            q = (test_counts + 1e-4) / (len(test_vals) + 1e-3)

            psi = float(np.sum((q - p) * np.log(q / p)))
            psi_results.append({
                "feature": col,
                "importance": int(importances[self.feature_cols.index(col)]),
                "psi": round(psi, 4),
                "drift_status": "High Drift" if psi > 0.25 else ("Moderate" if psi > 0.1 else "Stable")
            })
        return pd.DataFrame(psi_results)

    def check_leakage_and_duplicates(self) -> dict:
        """5 & 6. Audits exact duplicates and identical feature rows across splits."""
        train_ids = set(self.train_df["txId"])
        val_ids = set(self.val_df["txId"])
        test_ids = set(self.test_df["txId"])

        # Check overlapping transaction IDs
        id_overlap_train_val = len(train_ids.intersection(val_ids))
        id_overlap_val_test = len(val_ids.intersection(test_ids))

        # Check duplicate feature vectors (sampling first 20 columns for speed)
        cols_to_check = self.feature_cols[:20]
        train_dups = int(self.train_df.duplicated(subset=cols_to_check).sum())
        test_dups = int(self.test_df.duplicated(subset=cols_to_check).sum())

        return {
            "id_overlap_train_val": id_overlap_train_val,
            "id_overlap_val_test": id_overlap_val_test,
            "train_feature_duplicates": train_dups,
            "test_feature_duplicates": test_dups,
        }
