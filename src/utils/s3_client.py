import io
import boto3
from botocore.client import Config
import pandas as pd
from configs import settings

def get_s3_client():
    """Returns a boto3 S3 client configured for Garage path-style addressing."""
    return boto3.client(
        "s3",
        endpoint_url=settings.S3_ENDPOINT_URL,
        aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
        aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
        region_name=settings.AWS_DEFAULT_REGION,
        config=Config(s3={"addressing_style": "path"}),
    )

def upload_dataframe_to_parquet(
    df: pd.DataFrame,
    s3_key: str,
    bucket_name: str = settings.S3_BUCKET_NAME,
) -> str:
    """Uploads an in-memory Pandas DataFrame directly to Garage as a Parquet file."""
    client = get_s3_client()
    buffer = io.BytesIO()
    df.to_parquet(buffer, index=False, engine="pyarrow")
    buffer.seek(0)

    client.put_object(
        Bucket=bucket_name,
        Key=s3_key,
        Body=buffer.getvalue(),
    )
    return f"s3://{bucket_name}/{s3_key}"

def read_parquet_from_s3(
    s3_key: str,
    bucket_name: str = settings.S3_BUCKET_NAME,
) -> pd.DataFrame:
    """Reads a Parquet file from Garage directly into a Pandas DataFrame."""
    client = get_s3_client()
    response = client.get_object(Bucket=bucket_name, Key=s3_key)
    buffer = io.BytesIO(response["Body"].read())
    return pd.read_parquet(buffer, engine="pyarrow")

def list_s3_keys_with_prefix(
    prefix: str,
    bucket_name: str = settings.S3_BUCKET_NAME,
) -> list[str]:
    """Lists all object keys matching a prefix in the given bucket."""
    client = get_s3_client()
    paginator = client.get_paginator("list_objects_v2")
    keys = []
    for page in paginator.paginate(Bucket=bucket_name, Prefix=prefix):
        if "Contents" in page:
            keys.extend(obj["Key"] for obj in page["Contents"] if obj["Key"].endswith(".parquet"))
    return sorted(keys)
