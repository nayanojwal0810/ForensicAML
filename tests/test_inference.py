import pytest
import numpy as np
import pandas as pd
import joblib

from fastapi.testclient import TestClient
from app.main import app, service
from src.config import SCHEMA_PATH, FROZEN_THRESHOLD

client = TestClient(app)

@pytest.fixture(scope="module")
def schema():
    return joblib.load(SCHEMA_PATH)

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "online"
    assert "model_version" in data
    assert "schema_version" in data

def test_inference_validation_missing_feature(schema):
    payload = {
        "transaction_id": "tx_1",
        "features": {f: 0.0 for f in schema[:-1]} # Missing one
    }
    response = client.post("/v1/score", json=payload)
    assert response.status_code == 422
    assert "Missing 1 required features" in response.json()["detail"]

def test_inference_validation_unexpected_feature(schema):
    features = {f: 0.0 for f in schema}
    features["extra_feat"] = 1.0
    payload = {
        "transaction_id": "tx_1",
        "features": features
    }
    response = client.post("/v1/score", json=payload)
    assert response.status_code == 422
    assert "Unexpected 1 features provided" in response.json()["detail"]

def test_inference_validation_nan(schema):
    features = {f: 0.0 for f in schema}
    features[schema[0]] = "NaN"
    payload = {
        "transaction_id": "tx_1",
        "features": features
    }
    response = client.post("/v1/score", json=payload)
    assert response.status_code == 422
    assert "NaN values are strictly prohibited" in response.json()["detail"]

def test_score_endpoint_success(schema):
    features = {f: 0.0 for f in schema}
    payload = {
        "transaction_id": "tx_1",
        "features": features
    }
    response = client.post("/v1/score", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "calibrated_probability" in data
    assert "decision" in data
    assert "policy" in data
    assert "top_5_shap_contributors" in data
    assert len(data["top_5_shap_contributors"]) == 5

def test_explain_endpoint_success(schema):
    features = {f: 0.0 for f in schema}
    payload = {
        "transaction_id": "tx_1",
        "features": features
    }
    response = client.post("/v1/explain", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "top_10_shap_contributors" in data
    assert "relevant_graph_features" in data
    assert "relevant_temporal_features" in data
    assert len(data["top_10_shap_contributors"]) <= 10

def test_top_5_percent_queue_logic():
    # If probability >= FROZEN_THRESHOLD, decision is INVESTIGATE
    assert service.threshold == FROZEN_THRESHOLD
    
    # Mocking probability
    df = pd.DataFrame([{f: 0.0 for f in service.schema}])
    result = service.predict(df, explain=False)
    
    assert "decision" in result
    if result["calibrated_probability"] >= FROZEN_THRESHOLD:
        assert result["decision"] == "INVESTIGATE"
    else:
        assert result["decision"] == "PASS"

