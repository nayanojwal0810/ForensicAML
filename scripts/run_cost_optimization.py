"""
scripts/run_cost_optimization.py
End-to-end execution of Phase 4. Freezes threshold on validation and evaluates untouched test.
"""

import os
import joblib
import numpy as np
import pandas as pd
from scipy.special import logit

from src.ingestion.data_loader import EllipticDataLoader
from src.features.forensics.graph import TemporalGraphFeatureExtractor
from src.features.temporal import TemporalBehavioralFeatureExtractor
from src.evaluation.cost_optimizer import CostOptimizer
from src.evaluation.capacity_analysis import CapacityAnalyzer
from src.evaluation.cost_sensitivity import CostSensitivityAnalyzer

def main():
    print("[INFO] Initializing Phase 4: Business Cost Optimization...")
    os.makedirs("reports", exist_ok=True)

    # 1. Load Frozen Artifacts
    print("[INFO] Loading frozen artifacts and data...")
    lgb_model = joblib.load("artifacts/models/lightgbm_m2_pruned.joblib")
    platt_calibrator = joblib.load("artifacts/calibrators/platt_calibrator.joblib")
    frozen_feats = joblib.load("artifacts/models/feature_schema_v1.joblib")

    # Reconstruct exact splits used in M2 (in production, read from a Feature Store/Parquet)
    loader = EllipticDataLoader("data/raw")
    splits = loader.get_temporal_splits()
    df_graph = TemporalGraphFeatureExtractor(splits.full_nodes, splits.edges).extract_features()
    raw_feats, _, _ = loader.load_raw_data()
    df_temp = TemporalBehavioralFeatureExtractor(raw_feats).extract_features()
    
    val_m2 = pd.merge(pd.merge(splits.val, df_graph, on="txId", how="left"), df_temp, on="txId", how="left").fillna(0)
    test_m2 = pd.merge(pd.merge(splits.test, df_graph, on="txId", how="left"), df_temp, on="txId", how="left").fillna(0)

    y_va = val_m2["label"].to_numpy(dtype=int)
    X_va = val_m2[frozen_feats]
    y_te = test_m2["label"].to_numpy(dtype=int)
    X_te = test_m2[frozen_feats]

    # 2. Generate Calibrated Probabilities
    def get_calibrated_probs(X):
        raw_prob = lgb_model.predict_proba(X)[:, 1]
        clipped = np.clip(raw_prob, 1e-6, 1.0 - 1e-6)
        return platt_calibrator.predict_proba(logit(clipped).reshape(-1, 1))[:, 1]

    val_probs = get_calibrated_probs(X_va)
    test_probs = get_calibrated_probs(X_te)

    # 3. Configure Assumptions
    C_INV = 50.0
    C_LOSS = 2000.0

    print(f"[INFO] Assumptions Loaded: C_inv=${C_INV}, C_loss=${C_LOSS}")

    # 4. Validation Tuning
    print("[INFO] Optimizing decision policies on VALIDATION ONLY...")
    optimizer = CostOptimizer()
    val_optimal = optimizer.optimize_unconstrained(y_va, val_probs, C_INV, C_LOSS)
    FROZEN_THRESHOLD = val_optimal.threshold
    print(f"[INFO] FROZEN Optimal Threshold (from Val): {FROZEN_THRESHOLD:.4f}")

    # Capacity & Sensitivity on Validation
    cap_analyzer = CapacityAnalyzer(optimizer)
    df_cap_val = cap_analyzer.analyze_capacities(y_va, val_probs, [0.005, 0.01, 0.02, 0.05], C_INV, C_LOSS)

    sens_analyzer = CostSensitivityAnalyzer(optimizer)
    scenarios = [(50.0, 1000.0), (50.0, 2000.0), (50.0, 5000.0), (100.0, 2000.0)]
    df_sens_val = sens_analyzer.run_scenarios(y_va, val_probs, scenarios)

    # 5. Untouched Test Evaluation
    print("[INFO] Applying frozen policy to UNTOUCHED TEST SET...")
    total_illicit_te = int(np.sum(y_te == 1))
    baseline_loss_te = total_illicit_te * C_LOSS

    test_opt = optimizer.evaluate_threshold(y_te, test_probs, FROZEN_THRESHOLD, C_INV, C_LOSS, "Calibrated + Cost-Optimized")
    
    raw_te_probs = lgb_model.predict_proba(X_te)[:, 1]
    test_uncal = optimizer.evaluate_threshold(y_te, raw_te_probs, 0.5, C_INV, C_LOSS, "Uncalibrated M2 (t=0.5)")

    cost_reduction_pct = ((baseline_loss_te - test_opt.total_cost) / baseline_loss_te) * 100

    # 6. Generate Markdown Report
    print("[INFO] Generating final markdown report...")
    md_content = f"""# AML Surveillance: Business Cost Optimization Report

## 1. Explicit Assumptions & Constraints
* **Validation Period**: Time Steps 35–41.
* **Test Period**: Time Steps 42–49 (Untouched until final policy).
* **Investigation Cost ($C_{{inv}}$)**: ${C_INV:,.2f} per case.
* **Missed Illicit Loss ($C_{{loss}}$)**: ${C_LOSS:,.2f} flat exposure per missed transaction.

## 2. Frozen Operational Policy
* The decision threshold was optimized strictly on the Validation set.
* **FROZEN THRESHOLD**: `{FROZEN_THRESHOLD:.4f}`

## 3. Validation Analysis (No Test Information Used)

### A. Capacity Constraints (Top-K% Queues)
{df_cap_val.to_markdown(index=False)}

### B. Sensitivity Analysis (Cost Matrix Stress-Test)
{df_sens_val.to_markdown(index=False)}

## 4. Final Financial Outcome (Untouched Future Test)

| Policy | Threshold | Investigations | Recall (%) | Precision (%) | Total Expected Cost | Savings vs Baseline |
|---|---|---|---|---|---|---|
| Zero ML (Accept All) | - | 0 | 0.0% | 0.0% | ${baseline_loss_te:,.2f} | $0.00 |
| Uncalibrated M2 | 0.5000 | {test_uncal.investigations} | {test_uncal.recall*100:.1f}% | {test_uncal.precision*100:.1f}% | ${test_uncal.total_cost:,.2f} | ${(baseline_loss_te - test_uncal.total_cost):,.2f} |
| **Calibrated + Cost-Optimized** | **{test_opt.threshold:.4f}** | **{test_opt.investigations}** | **{test_opt.recall*100:.1f}%** | **{test_opt.precision*100:.1f}%** | **${test_opt.total_cost:,.2f}** | **${(baseline_loss_te - test_opt.total_cost):,.2f}** |

## 5. Executive Summary
* **Cost Reduction**: Achieved a **{cost_reduction_pct:.2f}%** reduction in expected loss vs taking no action.
* **Recall Multiplier**: Captured **{test_opt.recall*100:.1f}%** of illicit exposure in the strict future regime.
"""

    with open("reports/cost_optimization.md", "w") as f:
        f.write(md_content)

    print("[INFO] Execution complete. Report saved to reports/cost_optimization.md")
    print("[INFO] Recommendation: PROCEED TO PHASE 5")

if __name__ == "__main__":
    main()
