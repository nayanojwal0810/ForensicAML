"""
src/features/temporal.py
Point-in-time temporal behavioral feature extraction for AML surveillance.
"""

from __future__ import annotations
import numpy as np
import pandas as pd


class TemporalBehavioralFeatureExtractor:
    """Extracts causal, backward-looking temporal behavioral and regime features."""

    def __init__(self, full_features_df: pd.DataFrame) -> None:
        self.df = full_features_df[["txId", "time_step", "Local_feature_2", "fees", "size"]].copy()

    def extract_features(self) -> pd.DataFrame:
        """Computes backward-looking rolling aggregations strictly using time <= t."""
        print("[INFO] Computing point-in-time temporal behavioral features across 49 steps...")

        # 1. Step-level macro statistics (strictly historical: up to t)
        step_stats = (
            self.df.groupby("time_step")
            .agg(
                step_tx_count=("txId", "count"),
                step_fee_mean=("fees", "mean"),
                step_size_mean=("size", "mean"),
                step_loc2_mean=("Local_feature_2", "mean"),
            )
            .reset_index()
            .sort_values("time_step")
        )

        # 2. Backward-looking rolling windows (closed='left' to prevent within-step leakage)
        # Shift step stats by 1 step so that time t only sees information from t-1 and prior
        shifted_stats = step_stats.copy()
        shifted_stats["time_step"] = shifted_stats["time_step"] + 1  # maps stats of (t-1) to step t

        shifted_stats = shifted_stats.sort_values("time_step")
        
        # Rolling 3-step historical statistics (t-3 to t-1)
        shifted_stats["roll3_tx_count_mean"] = (
            shifted_stats["step_tx_count"].rolling(window=3, min_periods=1).mean()
        )
        shifted_stats["roll3_tx_count_std"] = (
            shifted_stats["step_tx_count"].rolling(window=3, min_periods=1).std().fillna(1.0)
        )
        shifted_stats["roll3_fee_mean"] = (
            shifted_stats["step_fee_mean"].rolling(window=3, min_periods=1).mean()
        )
        shifted_stats["roll3_size_mean"] = (
            shifted_stats["step_size_mean"].rolling(window=3, min_periods=1).mean()
        )
        shifted_stats["roll3_loc2_mean"] = (
            shifted_stats["step_loc2_mean"].rolling(window=3, min_periods=1).mean()
        )

        # EWMA of historical transaction volume (alpha = 0.5)
        shifted_stats["ewma_hist_tx_count"] = (
            shifted_stats["step_tx_count"].ewm(alpha=0.5, adjust=False).mean()
        )

        # Merge step-level macro metrics back into step_stats
        macro_df = pd.merge(
            step_stats[["time_step", "step_tx_count", "step_fee_mean", "step_size_mean"]],
            shifted_stats[
                [
                    "time_step",
                    "roll3_tx_count_mean",
                    "roll3_tx_count_std",
                    "roll3_fee_mean",
                    "roll3_size_mean",
                    "roll3_loc2_mean",
                    "ewma_hist_tx_count",
                ]
            ],
            on="time_step",
            how="left",
        )

        # Fill step 1 (no prior history) with current step values safely
        macro_df["roll3_tx_count_mean"] = macro_df["roll3_tx_count_mean"].fillna(macro_df["step_tx_count"])
        macro_df["roll3_fee_mean"] = macro_df["roll3_fee_mean"].fillna(macro_df["step_fee_mean"])
        macro_df["roll3_size_mean"] = macro_df["roll3_size_mean"].fillna(macro_df["step_size_mean"])
        macro_df["roll3_loc2_mean"] = macro_df["roll3_loc2_mean"].fillna(0.0)
        macro_df["ewma_hist_tx_count"] = macro_df["ewma_hist_tx_count"].fillna(macro_df["step_tx_count"])

        # 3. Macro burstiness and velocity indicators
        macro_df["tx_volume_velocity"] = (
            macro_df["step_tx_count"] / (macro_df["roll3_tx_count_mean"] + 1e-5)
        )
        macro_df["tx_burstiness_zscore"] = (
            (macro_df["step_tx_count"] - macro_df["roll3_tx_count_mean"])
            / (macro_df["roll3_tx_count_std"] + 1e-5)
        )
        macro_df["fee_velocity"] = (
            macro_df["step_fee_mean"] / (macro_df["roll3_fee_mean"] + 1e-5)
        )

        # 4. Merge macro features to transaction-level table
        tx_temporal = pd.merge(
            self.df[["txId", "time_step", "fees", "size", "Local_feature_2"]],
            macro_df[
                [
                    "time_step",
                    "tx_volume_velocity",
                    "tx_burstiness_zscore",
                    "fee_velocity",
                    "ewma_hist_tx_count",
                    "roll3_fee_mean",
                    "roll3_size_mean",
                    "roll3_loc2_mean",
                ]
            ],
            on="time_step",
            how="left",
        )

        # 5. Micro-to-Macro relative behavioral indicators
        tx_temporal["rel_fee_to_hist_regime"] = (
            tx_temporal["fees"] - tx_temporal["roll3_fee_mean"]
        )
        tx_temporal["rel_size_to_hist_regime"] = (
            tx_temporal["size"] - tx_temporal["roll3_size_mean"]
        )
        tx_temporal["rel_loc2_to_hist_regime"] = (
            tx_temporal["Local_feature_2"] - tx_temporal["roll3_loc2_mean"]
        )

        # Select only newly engineered features and txId
        engineered_cols = [
            "txId",
            "tx_volume_velocity",
            "tx_burstiness_zscore",
            "fee_velocity",
            "ewma_hist_tx_count",
            "rel_fee_to_hist_regime",
            "rel_size_to_hist_regime",
            "rel_loc2_to_hist_regime",
        ]

        print(f"[INFO] Temporal features extracted for {len(tx_temporal):,} transactions.")
        return tx_temporal[engineered_cols].copy()
