import argparse
from datetime import datetime, timezone
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from configs import settings
from src.data.validate_data import validate_dataframe
from src.utils.s3_client import list_s3_keys_with_prefix, read_parquet_from_s3

def load_training_dataset(limit_batches: int = 5) -> pd.DataFrame:
    """Discovers and concatenates the most recent validated raw batches from Garage."""
    keys = list_s3_keys_with_prefix("raw/")
    if not keys:
        raise FileNotFoundError(f"No raw data found in bucket '{settings.S3_BUCKET_NAME}'. Run data generation first.")

    selected_keys = keys[-limit_batches:]
    print(f"Loading {len(selected_keys)} batches for training: {selected_keys}")

    dfs = [read_parquet_from_s3(k) for k in selected_keys]
    combined_df = pd.concat(dfs, ignore_index=True)

    # Validate aggregated data contract
    is_valid, errors = validate_dataframe(combined_df)
    if not is_valid:
        raise ValueError(f"Training data validation failed: {errors[:5]}")

    return combined_df

def build_pipeline() -> Pipeline:
    """Builds an end-to-end scikit-learn pipeline with preprocessing and classifier."""
    numeric_features = ["tenure_months", "monthly_charges", "total_charges", "support_tickets"]
    categorical_features = ["contract_type", "internet_service"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_features),
        ]
    )

    clf = RandomForestClassifier(
        n_estimators=100,
        max_depth=6,
        random_state=settings.RANDOM_STATE,
        n_jobs=-1,
    )

    return Pipeline(steps=[("preprocessor", preprocessor), ("classifier", clf)])

def train_and_register_model(n_batches: int = 5):
    """Executes model training, logs to MLflow, and registers the artifact."""
    mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
    mlflow.set_experiment(settings.MLFLOW_EXPERIMENT_NAME)

    df = load_training_dataset(limit_batches=n_batches)

    # Feature/target split
    X = df.drop(columns=["customer_id", "churn"])
    y = df["churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=settings.RANDOM_STATE, stratify=y
    )

    pipeline = build_pipeline()

    with mlflow.start_run(run_name=f"churn-train-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}") as run:
        pipeline.fit(X_train, y_train)

        # Evaluation metrics
        y_pred = pipeline.predict(X_test)
        y_proba = pipeline.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        auc = roc_auc_score(y_test, y_proba)

        # Log parameters & metrics
        mlflow.log_params(
            {
                "model_type": "RandomForestClassifier",
                "n_estimators": 100,
                "max_depth": 6,
                "training_rows": len(X_train),
                "evaluation_rows": len(X_test),
                "source_batches_count": n_batches,
            }
        )
        mlflow.log_metrics(
            {
                "accuracy": acc,
                "precision": prec,
                "recall": rec,
                "f1_score": f1,
                "roc_auc": auc,
            }
        )

        signature = infer_signature(X_train, pipeline.predict(X_train))

        # Log and register model using the modern `name` parameter
        mlflow.sklearn.log_model(
            sk_model=pipeline,
            name="model",
            registered_model_name=settings.MLFLOW_MODEL_NAME,
            signature=signature,
            input_example=X_train.head(3),
            skops_trusted_types=["sklearn.tree._tree.Tree"],
        )

        print(f"MLflow Run ID: {run.info.run_id}")
        print(f"Metrics: Accuracy={acc:.4f}, Precision={prec:.4f}, Recall={rec:.4f}, F1={f1:.4f}, AUC={auc:.4f}")
        print(f"Model successfully registered under name: '{settings.MLFLOW_MODEL_NAME}'")

def main():
    parser = argparse.ArgumentParser(description="Train customer churn model and log to MLflow")
    parser.add_argument("--batches", type=int, default=5, help="Number of latest raw batches to train on")
    args = parser.parse_args()

    train_and_register_model(n_batches=args.batches)

if __name__ == "__main__":
    main()
