# Model Card

## Purpose
This model identifies illicit cryptocurrency transactions within a defined temporal and graph network context, optimizing for high precision within a strict operational capacity queue.

## Dataset
* **Source**: Publicly available Elliptic++ research dataset.
* **Characteristics**: Highly imbalanced, non-stationary temporal dynamics, graph structure (transaction networks).
* **Constraints**: Incomplete labels (large number of 'unknown' transactions which were excluded from training).

## Model
* **Algorithm**: LightGBM (Gradient Boosting Decision Tree).
* **Architecture**: Pruned tree structure to mitigate overfitting on temporal noise.
* **Features**: 190 total features.

## Features
Includes base transaction details, temporal behavioral features, and graph-based network forensics (e.g., in-degree, out-degree, PageRank, pass-through ratio).

## Calibration
Uses a Platt Scaling Calibrator (Logistic Regression) to ensure raw model scores reflect true probability estimates, aiding in robust business cost optimization.

## Decision Policy
* **Policy**: Fixed 5% Capacity Queue.
* **Mechanism**: Transactions are scored and calibrated; the top 5% highest-risk transactions (which implicitly corresponds to a probability threshold of 0.1526 on the validation set) are flagged for human investigation.

## Explainability
* **Method**: TreeSHAP.
* **Purpose**: SHAP explains *model behavior* and the mathematical contributors to a high risk score. It does NOT establish criminal intent or legal proof.

## Monitoring
Continuous drift monitoring is required to detect concept drift across time steps, triggering alerts if statistical thresholds are breached.

## Limitations
* Scenario-based cost assumptions ($50 investigation cost, $2000 loss) are illustrative.
* This is not regulatory validation or a production banking deployment.
* Effectiveness depends on the assumption that the graph and temporal topology remain structurally similar to the training period.

