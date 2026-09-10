# Final Validation Results

All results are measured on the strictly untouched out-of-time test set (Steps 42-49) using frozen artifacts.

## 1. Predictive Quality
* **Out-of-time PR-AUC**: 0.5526
* **Out-of-time ROC-AUC**: 0.8524

## 2. Calibration Quality
* **Brier Score**: 0.0269
* **Expected Calibration Error (ECE)**: 0.0232

## 3. Operational Queue (Top 5% Capacity)
* **Recall@5%**: 50.00%
* **Precision@5%**: 46.05%
* **Random 5% Recall Baseline**: 5.01%
* **Recall Multiplier**: 9.98x

## 4. Business Scenario (C_inv=$50, C_loss=$2000)
* **Scenario cost reduction**: 47.26%

## 5. System Latency
* **FastAPI `/v1/score` p50**: 37.92 ms
* **FastAPI `/v1/explain` p50**: 43.82 ms

