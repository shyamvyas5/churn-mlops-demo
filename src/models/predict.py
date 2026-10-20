import argparse
from datetime import datetime, timezone
import sys
import mlflow
import mlflow.sklearn
import pandas as pd
from configs import settings
from src.utils.s3_client import list_s3_keys_with_prefix, read_parquet_from_s3, upload_dataframe_to_parquet

def run_batch_scoring(input_s3_key: str | None = None) -> str:
    """Pulls an input batch from Garage, scores it via the MLflow model, and saves results."""
    mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
    model_uri = f"models:/{settings.MLFLOW_MODEL_NAME}/latest"

    print(f"Connecting to MLflow: {settings.MLFLOW_TRACKING_URI}")
    print(f"Loading registered model from: {model_uri}")
    try:
        # Load directly as a native scikit-learn Pipeline
        model = mlflow.sklearn.load_model(model_uri)
    except Exception as exc:
        print(f"Failed to load model from MLflow: {exc}", file=sys.stderr)
        sys.exit(1)

    # Determine input batch to score
    if not input_s3_key:
        raw_keys = list_s3_keys_with_prefix("raw/")
        if not raw_keys:
            print("No raw batches found to score.", file=sys.stderr)
            sys.exit(1)
        input_s3_key = raw_keys[-1]

    print(f"Scoring dataset: s3://{settings.S3_BUCKET_NAME}/{input_s3_key}")
    df = read_parquet_from_s3(input_s3_key)

    customer_ids = df["customer_id"].copy()
    features = df.drop(columns=["customer_id", "churn"], errors="ignore")

    # Generate predictions & probabilities directly from the scikit-learn pipeline
    preds = model.predict(features)

    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(features)[:, 1]
    else:
        probs = [0.0] * len(preds)

    scored_df = pd.DataFrame(
        {
            "customer_id": customer_ids,
            "churn_prediction": preds,
            "churn_probability": probs,
            "scored_at_utc": datetime.now(timezone.utc).isoformat(),
        }
    )

    # Destination key in Garage processed partition
    now = datetime.now(timezone.utc)
    ts_str = now.strftime("%Y%m%dT%H%M%SZ")
    output_key = f"processed/year={now.year}/month={now.month:02d}/day={now.day:02d}/predictions_{ts_str}.parquet"

    destination_uri = upload_dataframe_to_parquet(scored_df, s3_key=output_key)
    print(f"Batch scoring complete. {len(scored_df)} rows written to: {destination_uri}")
    return destination_uri

def main():
    parser = argparse.ArgumentParser(description="Run batch scoring on customer churn data")
    parser.add_argument("--s3-key", type=str, default=None, help="Specific S3 key to score")
    args = parser.parse_args()

    run_batch_scoring(input_s3_key=args.s3_key)

if __name__ == "__main__":
    main()
