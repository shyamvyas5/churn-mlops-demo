from fastapi.testclient import TestClient
from src.api.app import app, model_container

def test_healthz_endpoint():
    with TestClient(app) as client:
        response = client.get("/healthz")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "model_loaded" in data
        assert "model_name" in data

def test_predict_endpoint_empty_payload():
    with TestClient(app) as client:
        response = client.post("/predict", json={"customers": []})
        # If model is not loaded yet in test env, 503 is expected;
        # otherwise 400 for empty list
        assert response.status_code in [400, 503]

def test_predict_schema_validation_error():
    with TestClient(app) as client:
        # Invalid payload missing required fields like monthly_charges
        bad_payload = {
            "customers": [
                {
                    "customer_id": "CUST-999",
                    "tenure_months": 12,
                }
            ]
        }
        response = client.post("/predict", json=bad_payload)
        assert response.status_code == 422  # Unprocessable Entity (Pydantic failure)
