import argparse
from datetime import datetime, timezone
import uuid
import numpy as np
import pandas as pd
from configs import settings
from src.utils.s3_client import upload_dataframe_to_parquet

def generate_synthetic_churn_data(
    num_records: int = settings.BATCH_SIZE,
    seed: int = settings.RANDOM_STATE,
) -> pd.DataFrame:
    """Generates realistic synthetic SaaS churn data with statistical dependencies."""
    np.random.seed(seed)

    customer_ids = [f"CUST-{uuid.uuid4().hex[:8].upper()}" for _ in range(num_records)]
    tenure_months = np.random.randint(1, 72, size=num_records)
    contract_types = np.random.choice(
        ["Month-to-month", "One year", "Two year"],
        size=num_records,
        p=[0.55, 0.25, 0.20],
    )
    internet_services = np.random.choice(
        ["Fiber optic", "DSL", "No"],
        size=num_records,
        p=[0.45, 0.35, 0.20],
    )
    support_tickets = np.random.poisson(lam=1.5, size=num_records)

    # Base monthly charges influenced by internet service
    base_charges = {"Fiber optic": 75.0, "DSL": 45.0, "No": 20.0}
    monthly_charges = np.array([base_charges[svc] for svc in internet_services]) + np.random.uniform(
        0, 25, size=num_records
    )
    monthly_charges = np.round(monthly_charges, 2)

    # Total charges roughly equals tenure * monthly_charges + small noise
    total_charges = np.round(
        tenure_months * monthly_charges + np.random.normal(0, 10, size=num_records),
        2,
    )
    total_charges = np.clip(total_charges, a_min=0.0, a_max=None)

    # Calculate churn probability using a logistic sigmoid
    # Higher churn risk: month-to-month, high support tickets, short tenure
    logit = (
        -1.2
        - (0.05 * tenure_months)
        + (0.40 * (contract_types == "Month-to-month").astype(int))
        + (0.35 * support_tickets)
        + (0.01 * monthly_charges)
    )
    churn_probs = 1.0 / (1.0 + np.exp(-logit))
    churn = np.random.binomial(1, churn_probs)

    df = pd.DataFrame(
        {
            "customer_id": customer_ids,
            "tenure_months": tenure_months,
            "monthly_charges": monthly_charges,
            "total_charges": total_charges,
            "contract_type": contract_types,
            "internet_service": internet_services,
            "support_tickets": support_tickets,
            "churn": churn,
        }
    )
    return df

def main():
    parser = argparse.ArgumentParser(description="Generate synthetic churn batches")
    parser.add_argument("--size", type=int, default=settings.BATCH_SIZE, help="Number of records to generate")
    args = parser.parse_args()

    df = generate_synthetic_churn_data(num_records=args.size)

    # UTC timestamped partition path
    now = datetime.now(timezone.utc)
    ts_str = now.strftime("%Y%m%dT%H%M%SZ")
    s3_key = f"raw/year={now.year}/month={now.month:02d}/day={now.day:02d}/batch_{ts_str}.parquet"

    s3_uri = upload_dataframe_to_parquet(df, s3_key=s3_key)
    print(f"Successfully generated {len(df)} rows and uploaded to: {s3_uri}")

if __name__ == "__main__":
    main()
