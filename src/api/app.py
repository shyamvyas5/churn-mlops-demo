import logging
from contextlib import asynccontextmanager
import mlflow.sklearn
import pandas as pd
from fastapi import FastAPI, HTTPException, status
from configs import settings
from src.api.schemas import (
    CustomerFeatures,
    HealthResponse,
    PredictionRequest,
    PredictionResponse,
    PredictionResult,
)

logging.basicConfig(level=settings.LOG_LEVEL)
logger = logging.getLogger(settings.APP_NAME)

# Global model cache
model_container = {}

# Inside the lifespan context manager:
@asynccontextmanager
async def lifespan(app: FastAPI):
    mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
    model_uri = f"models:/{settings.MLFLOW_MODEL_NAME}/latest"
    logger.info(f"Loading registered model from: {model_uri}")
    try:
        model = mlflow.sklearn.load_model(model_uri)
        model_container["model"] = model
        model_container["version"] = "latest"
        logger.info("MLflow model loaded successfully into FastAPI.")
    except Exception as exc:
        logger.warning(f"Could not load registered model '{settings.MLFLOW_MODEL_NAME}': {exc}")
        model_container["model"] = None
        model_container["version"] = "none"

    yield
    model_container.clear()

app = FastAPI(
    title=settings.APP_NAME,
    description="Batch & Realtime Customer Churn Prediction Service",
    lifespan=lifespan,
)

@app.get("/healthz", response_model=HealthResponse, tags=["Monitoring"])
def healthcheck():
    return HealthResponse(
        status="healthy",
        model_loaded=model_container.get("model") is not None,
        model_name=settings.MLFLOW_MODEL_NAME,
    )

@app.post("/predict", response_model=PredictionResponse, tags=["Inference"])
def predict(request: PredictionRequest):
    model = model_container.get("model")
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not loaded. Train and register the model in MLflow first.",
        )

    if not request.customers:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Empty customer payload received.",
        )

    # Convert Pydantic records to DataFrame matching training format
    records = [c.model_dump() for c in request.customers]
    df = pd.DataFrame(records)
    customer_ids = df.pop("customer_id")

    try:
        # Generate predictions
        # MLflow pyfunc model returns numpy array or series
        predictions = model.predict(df)

        # Retrieve probabilities if supported by the underlying model
        probabilities = [0.0] * len(df)
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(df)[:, 1].tolist()
        elif hasattr(model.unwrap_python_model(), "predict_proba"):
            raw_estimator = model.unwrap_python_model()
            probabilities = raw_estimator.predict_proba(df)[:, 1].tolist()

        results = [
            PredictionResult(
                customer_id=cid,
                churn_prediction=int(pred),
                churn_probability=float(prob),
            )
            for cid, pred, prob in zip(customer_ids, predictions, probabilities)
        ]

        return PredictionResponse(
            model_version=model_container.get("version", "latest"),
            predictions=results,
        )
    except Exception as exc:
        logger.error(f"Inference error: {exc}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference execution failed: {str(exc)}",
        )
