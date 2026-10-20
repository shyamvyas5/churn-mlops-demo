import os
from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Application Settings
APP_NAME = os.getenv("APP_NAME", "churn-scoring-service")
APP_ENV = os.getenv("APP_ENV", "development")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# MLflow Settings
# When running locally outside K8s, port-forward 5000:5000 to use localhost
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
MLFLOW_EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "churn-batch-scoring")
MLFLOW_MODEL_NAME = os.getenv("MLFLOW_MODEL_NAME", "churn_classifier")

# S3 / Garage Settings
S3_ENDPOINT_URL = os.getenv("S3_ENDPOINT_URL", "http://localhost:3900")
S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME", "churn-mlops")
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID", "")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY", "")
AWS_DEFAULT_REGION = os.getenv("AWS_DEFAULT_REGION", "garage")

# Data Generation & Pipeline Thresholds
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "1000"))
RANDOM_STATE = int(os.getenv("RANDOM_STATE", "42"))
RETRAIN_THRESHOLD_ROWS = int(os.getenv("RETRAIN_THRESHOLD_ROWS", "5000"))

# FastAPI Serving
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))
