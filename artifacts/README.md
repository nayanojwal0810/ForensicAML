# Artifacts

This directory stores the frozen machine learning artifacts required by the FastAPI inference service.

## Artifacts

* `models/lightgbm_m2_pruned.joblib`: The validated, frozen LightGBM model. (Version: v1.0.0-m2-pruned)
* `calibrators/platt_calibrator.joblib`: The Platt calibrator fitted via Logistic Regression to align raw probabilities to actual expected prevalence. (Version: v1.0.0-platt)
* `models/feature_schema_v1.joblib`: The list of 190 validated features required for inference. Maintains exact ordering and naming. (Version: v1.0.0-feat-190)

These artifacts are final and frozen. They should not be regenerated or retrained for the provided evaluation results.

