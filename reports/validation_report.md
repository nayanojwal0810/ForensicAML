# Validation Report

## Validation Methodology
* **Temporal Split**: Strict chronological splitting to prevent future data leakage (Train: Steps 1-34, Validation: Steps 35-41, Test: Steps 42-49).
* **Leakage Checks**: Verified no overlap between train, validation, and test periods.
* **Graph Validation**: Confirmed correct network linkage. Included cycle and strongly connected component (SCC) finding for advanced feature engineering.

## Feature Selection
* **Graph Feature Ablation**: Graph forensics (PageRank, degrees) provided a measurable but modest incremental predictive lift over base features.
* **Temporal Feature Selection**: Evaluated the inclusion of rolling behavioral profiles.
* **Final Schema**: 190 total features retained.

## Model Performance & Stability
* **Calibration Comparison**: Platt scaling successfully smoothed extreme probabilities, yielding lower Brier and ECE scores.
* **Temporal Drift Findings**: The underlying graph and transaction patterns demonstrate non-stationarity over time. While the pruned LightGBM generalized reasonably well out-of-time, continuous monitoring is advised due to observed mild temporal drift.

