# Cost Analysis

## Business Scenario
A hypothetical investigation pipeline evaluates cost-efficiency versus raw predictive power.

**Assumptions:**
* $C_{inv}$ (Cost per Investigation): $50.00
* $C_{loss}$ (Cost of Missed Illicit Transaction): $2000.00

*Note: These scenario assumptions are distinct from observed statistical measurements.*

## Policy Evaluation
An unconstrained optimization (threshold = 0.5) was rejected due to overwhelming alert volumes that exceed operational capacity. Instead, a strict resource-constrained policy was selected.

* **Chosen Policy**: Top 5% Fixed-Capacity Queue.
* **Implicit Threshold**: 0.1526 (established purely on the validation set).
* **Random 5% Baseline**: Randomly reviewing 5% of transactions yields an expected 5.01% recall.

## Final Test Outcomes
Applying the frozen 0.1526 threshold to the untouched test set (Steps 42-49):
* **Recall Multiplier**: 9.98x improvement over the random 5% baseline.
* **Cost Reduction**: Expected overall financial loss is reduced by 47.26% compared to taking no action.

## Sensitivity
Varying the cost parameters ($C_{inv}$ ranging $50-$100, $C_{loss}$ ranging $1000-$5000) confirms the robustness of the chosen threshold for preserving positive ROI within reasonable bounds.

