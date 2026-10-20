import pandas as pd
import pytest
from src.data.validate_data import validate_dataframe

@pytest.fixture
def valid_churn_record():
    return {
        "customer_id": "CUST-A1B2C3D4",
        "tenure_months": 24,
        "monthly_charges": 65.50,
        "total_charges": 1572.00,
        "contract_type": "One year",
        "internet_service": "Fiber optic",
        "support_tickets": 2,
        "churn": 0,
    }

def test_valid_dataframe_passes(valid_churn_record):
    df = pd.DataFrame([valid_churn_record])
    is_valid, errors = validate_dataframe(df)
    assert is_valid is True
    assert len(errors) == 0

def test_null_value_detected(valid_churn_record):
    record = valid_churn_record.copy()
    record["monthly_charges"] = None
    df = pd.DataFrame([record])

    is_valid, errors = validate_dataframe(df)
    assert is_valid is False
    assert any("contains 1 null value" in err for err in errors)

def test_invalid_contract_type_detected(valid_churn_record):
    record = valid_churn_record.copy()
    record["contract_type"] = "Five year"  # Not a valid ContractType enum
    df = pd.DataFrame([record])

    is_valid, errors = validate_dataframe(df)
    assert is_valid is False
    assert any("contract_type" in err for err in errors)

def test_out_of_bound_tenure_detected(valid_churn_record):
    record = valid_churn_record.copy()
    record["tenure_months"] = 250  # Boundary violation (le=120)
    df = pd.DataFrame([record])

    is_valid, errors = validate_dataframe(df)
    assert is_valid is False
    assert any("tenure_months" in err for err in errors)

def test_invalid_churn_label_detected(valid_churn_record):
    record = valid_churn_record.copy()
    record["churn"] = 5  # Must be 0 or 1
    df = pd.DataFrame([record])

    is_valid, errors = validate_dataframe(df)
    assert is_valid is False
    assert any("churn" in err for err in errors)
