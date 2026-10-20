from enum import Enum
from typing import List
from pydantic import BaseModel, Field


class ContractType(str, Enum):
    MONTH_TO_MONTH = "Month-to-month"
    ONE_YEAR = "One year"
    TWO_YEAR = "Two year"

class InternetService(str, Enum):
    DSL = "DSL"
    FIBER_OPTIC = "Fiber optic"
    NO = "No"

class CustomerFeatures(BaseModel):
    customer_id: str = Field(..., description="Unique customer identifier")
    tenure_months: int = Field(..., ge=0, le=120, description="Tenure in months")
    monthly_charges: float = Field(..., ge=0.0, le=1000.0, description="Monthly subscription charges")
    total_charges: float = Field(..., ge=0.0, description="Total charges accumulated")
    contract_type: ContractType = Field(..., description="Customer contract commitment")
    internet_service: InternetService = Field(..., description="Internet service type")
    support_tickets: int = Field(default=0, ge=0, le=50, description="Number of support tickets logged")

    class Config:
        use_enum_values = True

class BatchValidationRecord(CustomerFeatures):
    # Includes target label for training/evaluation batches
    churn: int = Field(..., ge=0, le=1, description="Ground truth churn flag (0 or 1)")

class PredictionRequest(BaseModel):
    customers: List[CustomerFeatures]

class PredictionResult(BaseModel):
    customer_id: str
    churn_prediction: int = Field(..., description="Predicted churn: 0 (Stay) or 1 (Churn)")
    churn_probability: float = Field(..., ge=0.0, le=1.0, description="Predicted probability of churn")

class PredictionResponse(BaseModel):
    model_version: str
    predictions: List[PredictionResult]

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_name: str
