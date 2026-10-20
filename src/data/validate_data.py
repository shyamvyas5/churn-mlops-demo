import argparse
import sys
import pandas as pd
from pydantic import ValidationError
from configs import settings
from src.api.schemas import BatchValidationRecord
from src.utils.s3_client import list_s3_keys_with_prefix, read_parquet_from_s3

def validate_dataframe(df: pd.DataFrame) -> tuple[bool, list[str]]:
    """Validates dataframe schema and boundary conditions using Pydantic BatchValidationRecord.

    Returns:
        (is_valid, list_of_error_messages)
    """
    errors = []

    # 1. Null check across DataFrame
    null_counts = df.isnull().sum()
    columns_with_nulls = null_counts[null_counts > 0]
    if not columns_with_nulls.empty:
        for col, count in columns_with_nulls.items():
            errors.append(f"Column '{col}' contains {count} null value(s).")

    # 2. Record-level schema validation via Pydantic
    records = df.to_dict(orient="records")
    for idx, record in enumerate(records):
        try:
            BatchValidationRecord.model_validate(record)
        except ValidationError as err:
            for error_detail in err.errors():
                loc = ".".join(str(l) for l in error_detail.get("loc", []))
                msg = error_detail.get("msg", "Validation failed")
                errors.append(f"Row {idx} invalid for field '{loc}': {msg}")
            # Cap reported errors to avoid terminal flooding
            if len(errors) >= 50:
                errors.append("Exceeded maximum reportable validation errors (50). Truncating output.")
                break

    is_valid = len(errors) == 0
    return is_valid, errors

def load_and_validate_s3_key(s3_key: str) -> pd.DataFrame:
    """Reads a Parquet file from Garage, validates it, and returns the DataFrame or exits with code 1."""
    print(f"Validating file: s3://{settings.S3_BUCKET_NAME}/{s3_key}")
    df = read_parquet_from_s3(s3_key)

    is_valid, errors = validate_dataframe(df)
    if not is_valid:
        print("\n[VALIDATION FAILED] Data quality checks failed with the following errors:", file=sys.stderr)
        for err in errors:
            print(f"  - {err}", file=sys.stderr)
        sys.exit(1)

    print(f"[VALIDATION PASSED] {len(df)} records verified against Pydantic schema.")
    return df

def main():
    parser = argparse.ArgumentParser(description="Validate Parquet batches from Garage S3")
    parser.add_argument("--s3-key", type=str, default=None, help="Specific S3 key to validate")
    args = parser.parse_args()

    target_key = args.s3_key
    if not target_key:
        keys = list_s3_keys_with_prefix("raw/")
        if not keys:
            print("No raw data files found in Garage bucket to validate.", file=sys.stderr)
            sys.exit(1)
        target_key = keys[-1]  # Pick the latest timestamped batch

    load_and_validate_s3_key(target_key)

if __name__ == "__main__":
    main()
