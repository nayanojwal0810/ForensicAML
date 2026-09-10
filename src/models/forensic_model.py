"""
src/models/forensic_model.py
M1: LightGBM enriched with point-in-time graph forensic features.
"""

from __future__ import annotations
from typing import Tuple
import lightgbm as lgb
import numpy as np
import pandas as pd


class ForensicClassifier:
    """Enriched classifier incorporating graph-topology forensic features (M1)."""

    def __init__(self, random_state: int = 42) -> None:
        self.random_state = random_state
        self.model: lgb.LGBMClassifier | None = None
        self.feature_names: list[str] = []

    def _prepare_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        ignore_cols = ["txId", "time_step", "label"]
        feature_cols = [c for c in df.columns if c not in ignore_cols]
        self.feature_names = feature_cols
        return df[feature_cols].to_numpy(dtype=np.float32), df["label"].to_numpy(dtype=int)

    def train(self, train_df: pd.DataFrame, val_df: pd.DataFrame) -> None:
        X_train, y_train = self._prepare_data(train_df)
        X_val, y_val = self._prepare_data(val_df)

        neg_count = np.sum(y_train == 0)
        pos_count = np.sum(y_train == 1)
        scale_pos_weight = float(neg_count / pos_count)

        self.model = lgb.LGBMClassifier(
            n_estimators=1000,
            learning_rate=0.03,
            num_leaves=63,
            min_child_samples=50,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            random_state=self.random_state,
            n_jobs=-1,
            verbosity=-1,
        )

        self.model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)],
        )

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model has not been trained.")
        X, _ = self._prepare_data(df)
        return self.model.predict_proba(X)[:, 1]
