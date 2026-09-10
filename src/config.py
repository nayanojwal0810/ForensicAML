"""
src/config.py
Centralized configuration for ForensicAML project.
"""

from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = BASE_DIR / "artifacts"
MODELS_DIR = ARTIFACTS_DIR / "models"
CALIBRATORS_DIR = ARTIFACTS_DIR / "calibrators"

# Artifact Paths
MODEL_PATH = MODELS_DIR / "lightgbm_m2_pruned.joblib"
CALIBRATOR_PATH = CALIBRATORS_DIR / "platt_calibrator.joblib"
SCHEMA_PATH = MODELS_DIR / "feature_schema_v1.joblib"

# Versions
MODEL_VERSION = "v1.0.0-m2-pruned"
SCHEMA_VERSION = "v1.0.0-feat-190"
CALIBRATOR_VERSION = "v1.0.0-platt"

# Business Policy
INVESTIGATION_CAPACITY_PCT = 0.05
FROZEN_THRESHOLD = 0.1526
C_INVESTIGATION = 50.0
C_LOSS = 2000.0

# Monitoring Settings
DRIFT_P_VALUE_THRESHOLD = 0.05

