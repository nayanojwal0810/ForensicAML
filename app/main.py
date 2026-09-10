"""
app/main.py
FastAPI application for AML surveillance inference and explainability.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Dict, Any, List
import pandas as pd
import numpy as np
import joblib

from src.inference.aml_service import AmlInferenceService
from src.config import (
    MODEL_PATH, CALIBRATOR_PATH, SCHEMA_PATH,
    FROZEN_THRESHOLD, MODEL_VERSION, CALIBRATOR_VERSION, SCHEMA_VERSION
)

# 1. Load Frozen Artifacts
try:
    lgb_model = joblib.load(MODEL_PATH)
    platt_calibrator = joblib.load(CALIBRATOR_PATH)
    frozen_feats = joblib.load(SCHEMA_PATH)
except FileNotFoundError:
    raise RuntimeError("Frozen artifacts not found.")

service = AmlInferenceService(
    model=lgb_model,
    calibrator=platt_calibrator,
    feature_schema=frozen_feats,
    investigation_threshold=FROZEN_THRESHOLD
)

app = FastAPI(title="AML Forensic Surveillance API", version="1.0.0")

# --- SCHEMAS ---

class TransactionPayload(BaseModel):
    transaction_id: str = Field(..., description="Unique transaction identifier")
    features: Dict[str, float] = Field(..., description="Exact 190 frozen feature key-value pairs")

# --- ENDPOINTS ---

@app.get("/health")
def health_check() -> Dict[str, str]:
    return {
        "status": "online",
        "model_version": MODEL_VERSION,
        "calibrator_version": CALIBRATOR_VERSION,
        "schema_version": SCHEMA_VERSION,
        "policy_version": f"Top-5% Capacity Queue (Threshold: {FROZEN_THRESHOLD})"
    }

def _process_request(payload: TransactionPayload) -> pd.DataFrame:
    """Validates input and converts to DataFrame for the service."""
    provided_keys = set(payload.features.keys())
    required_keys = set(service.schema)
    
    missing = required_keys - provided_keys
    unexpected = provided_keys - required_keys
    
    if missing:
        raise HTTPException(status_code=422, detail=f"Missing {len(missing)} required features.")
    if unexpected:
        raise HTTPException(status_code=422, detail=f"Unexpected {len(unexpected)} features provided.")
        
    df = pd.DataFrame([payload.features])[service.schema]
    
    if df.isna().any().any():
        raise HTTPException(status_code=422, detail="NaN values are strictly prohibited.")
    if np.isinf(df.to_numpy()).any():
        raise HTTPException(status_code=422, detail="Infinity values are strictly prohibited.")
        
    return df

@app.post("/v1/score")
def score_transaction(payload: TransactionPayload) -> Dict[str, Any]:
    df = _process_request(payload)
    
    # Generate prediction and Top-5 SHAP
    result = service.predict(df, explain=True)
    
    return {
        "transaction_id": payload.transaction_id,
        "calibrated_probability": result["calibrated_probability"],
        "decision": result["decision"],
        "policy": result["policy"],
        "top_5_shap_contributors": result["explanation"]["top_evidence"],
        "model_version": result["model_version"],
        "schema_version": result["schema_version"],
        "calibrator_version": result["calibrator_version"]
    }

@app.post("/v1/explain")
def explain_transaction(payload: TransactionPayload) -> Dict[str, Any]:
    df = _process_request(payload)
    
    # Extract prediction and Top-10 SHAP
    result = service.predict(df, explain=True)
    top_10_evidence = service.explainer.explain(df, top_k=10)["top_evidence"]
    
    graph_indicators = ["in_degree", "out_degree", "pagerank", "deg_imbalance", "pass_through_ratio"]
    temporal_indicators = ["rel_fee_to_hist_regime", "rel_size_to_hist_regime", "rel_loc2_to_hist_regime"]
    
    # Filter SHAP evidence for domain-specific indicators
    graph_evidence = [e for e in top_10_evidence if e["feature"] in graph_indicators]
    temporal_evidence = [e for e in top_10_evidence if e["feature"] in temporal_indicators]
    
    return {
        "transaction_id": payload.transaction_id,
        "calibrated_probability": result["calibrated_probability"],
        "decision": result["decision"],
        "top_10_shap_contributors": top_10_evidence,
        "relevant_graph_features": graph_evidence,
        "relevant_temporal_features": temporal_evidence
    }

