# Churn Scoring MLOps Demo

A simple containerized, Kubernetes-native MLOps demo for customer churn prediction model. The system automates synthetic data generation, Pydantic schema validation, model training and registration via MLflow, distributed storage in Garage S3, scheduled batch scoring, and real-time inference through a FastAPI service.

---

## What does this do?
In simple terms, this project helps a company **stop customers from leaving**.

* **The Problem:** Customers quietly cancel their subscriptions.
* **The Solution:** A machine learning model looks at customer behavior (contract type, monthly bill, tenure, and customer support complaints) and flags who is about to churn.
* **Impact:** Customer can be proactively offered discounts or their issues gets resolved for high-risk customers *before* they leave.

---

## How It Works (Step-by-Step)
1. **Synthetic Data Ingestion:** Generates realistic customer account and billing events and stores them as Parquet files in **Garage S3** storage.
2. **Data Validation:** Checks data types and ranges with **Pydantic** before the model ever sees them, preventing bad data from breaking downstream pipelines.
3. **Training & Registry:** Trains a **Random Forest** classification model, tracks accuracy and ROC-AUC metrics in **PostgreSQL**, and registers the model in **MLflow**.
4. **Batch Prediction:** Runs on a scheduled **Kubernetes CronJob** every 6 hours, processes thousands of customer accounts, and writes churn probabilities (`0.0` to `1.0`) back to Garage S3.
5. **Real-time Serving:** Exposes a high-speed **FastAPI** HTTP endpoint (`/predict`) on Kubernetes so support agents or customer dashboards can get instant churn risk scores on demand.

---

## Architecture & Technology Stack

* **Compute & Orchestration:** Kubernetes (`Job`, `CronJob`, `Deployment`, `Service`).
* **Experiment Tracking & Model Registry:** MLflow backed by PostgreSQL and Garage S3.
* **Object Storage:** Garage S3 (self-hosted S3-compatible distributed store) for raw datasets, model artifacts, and processed predictions.
* **Model Framework & Serialization:** Scikit-learn (`Pipeline`, `RandomForestClassifier`) serialized using `skops`.
* **Validation & Contracts:** Pydantic v2 validating ingestion schemas, ranges, null values, and API payloads.
* **Serving Layer:** FastAPI running on Uvicorn with structured health checks and lifespan model caching.

---

## Directory Structure
```text
.
├── configs/
│   └── settings.py              # Central Pydantic BaseSettings loading configs & envs
├── src/
│   ├── api/
│   │   ├── app.py               # FastAPI application with /healthz and /predict endpoints
│   │   └── schemas.py           # Pydantic request/response payload contracts
│   ├── data/
│   │   ├── generate_data.py     # Synthetic customer batch generator uploading to S3
│   │   └── validate_data.py     # Ingestion contract validator
│   ├── models/
│   │   ├── train.py             # Pulls raw batches, trains pipeline, logs/registers in MLflow
│   │   └── predict.py           # Pulls latest model, runs batch scoring, writes to S3
│   └── utils/
│       └── s3_client.py         # Boto3 client wrapper for Garage S3 operations
├── tests/
│   ├── test_data_validation.py  # Unit tests for data schema integrity
│   └── test_api.py              # Unit tests for FastAPI endpoints
├── Dockerfile                   # Multi-stage container definition
├── requirements.txt             # Python runtime dependencies
├── k8s
    ├── churn-mlops-deploy
    │   ├── api-serve-deploy.yaml  # Serve API
    │   ├── batch-predict-job.yaml # Batch prediction k8s job
    │   ├── generate-data-job.yaml # Generate fake data k8s job
    │   ├── configmap.yaml         # Configmap
    │   ├── secret.yaml            # Secret
    │   └── train-model-job.yaml   # Train model k8s job
    ├── garage
    │   ├── garage-bootstrap.sh    # Create bucket and all
    │   ├── garage-secret.yaml     # Garage secret
    │   └── garage-values.yaml     # Garage helm values override file
    ├── mlflow
    │   └── mlflow-values.yaml     # MLFlow helm values override file
    └── postgresql
        ├── postgresql-secret.yaml # PostgreSQL secret
        └── postgres-values.yaml   # PostgreSQL helm values override file
```

---
## Configuration & Environment Variables
| Variable | Description | Local Value (Port-Forward) | In-Cluster Value (K8s Service DNS) |
| :--- | :--- | :--- | :--- |
| `MLFLOW_TRACKING_URI` | MLflow HTTP Tracking URL | `http://localhost:5000` | `http://mlflow.mlops.svc.cluster.local:5000` |
| `MLFLOW_EXPERIMENT_NAME`| MLflow Experiment group | `churn-batch-scoring` | `churn-batch-scoring` |
| `MLFLOW_MODEL_NAME` | Registered model name | `churn_classifier` | `churn_classifier` |
| `S3_ENDPOINT_URL` | Object store endpoint | `http://localhost:3900` | `http://garage.mlops.svc.cluster.local:3900` |
| `MLFLOW_S3_ENDPOINT_URL`| MLflow S3 backend endpoint | `http://localhost:3900` | `http://garage.mlops.svc.cluster.local:3900` |
| `S3_BUCKET_NAME` | Garage S3 Bucket | `churn-mlops` | `churn-mlops` |
| `AWS_ACCESS_KEY_ID` | S3 Access Key | `` | *Loaded from Secret* |
| `AWS_SECRET_ACCESS_KEY` | S3 Secret Key | `` | *Loaded from Secret* |
| `AWS_DEFAULT_REGION` | S3 Region | `garage` | `garage` |
| `AWS_S3_FORCE_PATH_STYLE`| Path-style addressing flag | `"true"` | `"true"` |
| `MLFLOW_S3_IGNORE_TLS` | Bypass TLS verification | `"true"` | `"true"` |
| `GIT_PYTHON_REFRESH` | Silences Git warning logs | `quiet` | `quiet` |

---

## Local Development & Testing
### 1. Port-Forward Kubernetes Services
Forward MLflow and Garage to your local machine:
```bash
kubectl -n mlops port-forward svc/mlflow 5000:5000 &
kubectl -n mlops port-forward svc/garage 3900:3900 &
```

### 2. Set Up Virtual Environment
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export PYTHONPATH=.
```

### 3. Run Unit Tests
```bash
pytest tests/ -v
```

**Output Example:**
```text
============================= test session starts ==============================
collected 7 items

tests/test_data_validation.py::test_valid_dataframe_passes PASSED         [ 14%]
tests/test_data_validation.py::test_null_value_detected PASSED            [ 28%]
tests/test_data_validation.py::test_invalid_contract_type_detected PASSED [ 42%]
tests/test_data_validation.py::test_out_of_bound_tenure_detected PASSED   [ 57%]
tests/test_data_validation.py::test_invalid_churn_label_detected PASSED   [ 71%]
tests/test_api.py::test_healthz_endpoint PASSED                           [ 85%]
tests/test_api.py::test_predict_schema_validation_error PASSED            [100%]

============================== 7 passed in 0.82s ===============================
```

### 4. Execute Pipeline Locally
Export local environment variables:

```bash
export MLFLOW_TRACKING_URI="http://localhost:5000"
export MLFLOW_EXPERIMENT_NAME="churn-batch-scoring"
export MLFLOW_MODEL_NAME="churn_classifier"
export S3_ENDPOINT_URL="http://localhost:3900"
export S3_BUCKET_NAME="churn-mlops"
export MLFLOW_S3_ENDPOINT_URL="http://localhost:3900"
export AWS_ACCESS_KEY_ID="" # Put value here
export AWS_SECRET_ACCESS_KEY="" # Put value here
export AWS_DEFAULT_REGION="garage"
export AWS_S3_FORCE_PATH_STYLE="true"
export MLFLOW_S3_IGNORE_TLS="true"
export GIT_PYTHON_REFRESH="quiet"
```

Generate synthetic data batches:
```bash
python src/data/generate_data.py --size 1000
```
**Output Example:**
```text
Successfully generated 1000 rows and uploaded to: s3://churn-mlops/raw/year=2026/month=10/day=19/batch_20261019T054212Z.parquet
```

Train model and log to MLflow:
```bash
python src/models/train.py --batches 3
```
**Output Example:**
```text
Loading 3 batches for training: ['raw/.../batch_1.parquet', 'raw/.../batch_2.parquet', 'raw/.../batch_3.parquet']
MLflow Run ID: 23a46b84fdea42d6bf8e03b46881a578
Metrics: Accuracy=0.7873, Precision=0.5888, Recall=0.2490, F1=0.3500, AUC=0.7905
Model successfully registered under name: 'churn_classifier'
```

Run batch prediction locally:
```bash
python src/models/predict.py
```
**Output Example:**
```text
Connecting to MLflow: http://localhost:5000
Loading registered model from: models:/churn_classifier/latest
Scoring dataset: s3://churn-mlops/raw/year=2026/month=10/day=19/batch_20261019T054212Z.parquet
Batch scoring complete. 1000 rows written to: s3://churn-mlops/processed/year=2026/month=10/day=19/predictions_20261019T074122Z.parquet
```
Start Uvicorn server locally:
```bash
uvicorn src.api.app:app --host 0.0.0.0 --port 8000 --reload
```

**Expected Startup Logs:**

```text
INFO:     Started server process [12345]
INFO:     Waiting for application startup.
INFO:     src.api.app: Loading registered model from: models:/churn_classifier/latest
INFO:     src.api.app: MLflow model loaded successfully into FastAPI.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
```

**Test with curl:**
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "customers": [
      {
        "customer_id": "CUST-LOCAL-001",
        "tenure_months": 2,
        "monthly_charges": 85.50,
        "total_charges": 171.00,
        "contract_type": "Month-to-month",
        "internet_service": "Fiber optic",
        "support_tickets": 4
      }
    ]
  }'
```

**Output:**
```json
{
  "model_version": "latest",
  "predictions": [
    {
      "customer_id": "CUST-LOCAL-001",
      "churn_prediction": 1,
      "churn_probability": 0.7489693208669992
    }
  ]
}
```
