"""
src/ingestion/data_loader.py
Modular ingestion and leakage-free chronological splitting for Elliptic++.
"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple
import pandas as pd


@dataclass(frozen=True)
class DatasetSplits:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame
    edges: pd.DataFrame
    full_nodes: pd.DataFrame


class EllipticDataLoader:
    """Handles raw data ingestion, label normalization, and temporal partitioning."""

    def __init__(
        self,
        data_dir: str | Path = "data/raw",
        train_steps: Tuple[int, int] = (1, 34),
        val_steps: Tuple[int, int] = (35, 41),
        test_steps: Tuple[int, int] = (42, 49),
    ) -> None:
        self.data_dir = Path(data_dir)
        self.train_steps = train_steps
        self.val_steps = val_steps
        self.test_steps = test_steps

    def load_raw_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Loads raw CSVs and normalizes column names."""
        files = {f.name: f for f in self.data_dir.rglob("txs_*.csv")}
        
        df_classes = pd.read_csv(files["txs_classes.csv"])
        df_features = pd.read_csv(files["txs_features.csv"])
        df_edges = pd.read_csv(files["txs_edgelist.csv"])

        # Normalize column naming
        df_classes.columns = ["txId", "label"]
        df_edges.columns = ["source", "target"]
        
        # Standardize 'Time step' column to 'time_step'
        df_features.rename(columns={"Time step": "time_step"}, inplace=True)

        # Map labels: 1 -> 1 (illicit), 2 -> 0 (licit), 3 -> -1 (unknown)
        label_map = {1: 1, 2: 0, 3: -1}
        df_classes["label"] = df_classes["label"].map(label_map)

        return df_features, df_classes, df_edges

    def get_temporal_splits(self) -> DatasetSplits:
        """Merges features with labels and creates strict chronological splits."""
        features, classes, edges = self.load_raw_data()

        # Merge features and ground-truth labels
        df_full = pd.merge(features, classes, on="txId", how="inner")

        # Segregate labeled subset for supervised training/evaluation
        df_labeled = df_full[df_full["label"].isin([0, 1])].copy()

        # Chronological slices
        train = df_labeled[
            (df_labeled["time_step"] >= self.train_steps[0])
            & (df_labeled["time_step"] <= self.train_steps[1])
        ].copy()

        val = df_labeled[
            (df_labeled["time_step"] >= self.val_steps[0])
            & (df_labeled["time_step"] <= self.val_steps[1])
        ].copy()

        test = df_labeled[
            (df_labeled["time_step"] >= self.test_steps[0])
            & (df_labeled["time_step"] <= self.test_steps[1])
        ].copy()

        # Sanity Gate: Leakage assertions
        assert train["time_step"].max() < val["time_step"].min(), "[ERROR] Train/Val temporal overlap!"
        assert val["time_step"].max() < test["time_step"].min(), "[ERROR] Val/Test temporal overlap!"
        assert not train["label"].isin([-1]).any(), "[ERROR] Unknowns present in train set!"

        return DatasetSplits(
            train=train,
            val=val,
            test=test,
            edges=edges,
            full_nodes=df_full[["txId", "time_step", "label"]].copy(),
        )
