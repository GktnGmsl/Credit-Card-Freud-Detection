"""
Pydantic schemas for Credit Card Fraud Detection API.
"""

from pydantic import BaseModel, Field


class TransactionFeatures(BaseModel):
    """Raw transaction features — V1-V28 (PCA) + Time + Amount."""

    V1: float = Field(..., ge=-200, le=200, description="PCA component 1")
    V2: float = Field(..., ge=-200, le=200, description="PCA component 2")
    V3: float = Field(..., ge=-200, le=200, description="PCA component 3")
    V4: float = Field(..., ge=-200, le=200, description="PCA component 4")
    V5: float = Field(..., ge=-200, le=200, description="PCA component 5")
    V6: float = Field(..., ge=-200, le=200, description="PCA component 6")
    V7: float = Field(..., ge=-200, le=200, description="PCA component 7")
    V8: float = Field(..., ge=-200, le=200, description="PCA component 8")
    V9: float = Field(..., ge=-200, le=200, description="PCA component 9")
    V10: float = Field(..., ge=-200, le=200, description="PCA component 10")
    V11: float = Field(..., ge=-200, le=200, description="PCA component 11")
    V12: float = Field(..., ge=-200, le=200, description="PCA component 12")
    V13: float = Field(..., ge=-200, le=200, description="PCA component 13")
    V14: float = Field(..., ge=-200, le=200, description="PCA component 14")
    V15: float = Field(..., ge=-200, le=200, description="PCA component 15")
    V16: float = Field(..., ge=-200, le=200, description="PCA component 16")
    V17: float = Field(..., ge=-200, le=200, description="PCA component 17")
    V18: float = Field(..., ge=-200, le=200, description="PCA component 18")
    V19: float = Field(..., ge=-200, le=200, description="PCA component 19")
    V20: float = Field(..., ge=-200, le=200, description="PCA component 20")
    V21: float = Field(..., ge=-200, le=200, description="PCA component 21")
    V22: float = Field(..., ge=-200, le=200, description="PCA component 22")
    V23: float = Field(..., ge=-200, le=200, description="PCA component 23")
    V24: float = Field(..., ge=-200, le=200, description="PCA component 24")
    V25: float = Field(..., ge=-200, le=200, description="PCA component 25")
    V26: float = Field(..., ge=-200, le=200, description="PCA component 26")
    V27: float = Field(..., ge=-200, le=200, description="PCA component 27")
    V28: float = Field(..., ge=-200, le=200, description="PCA component 28")
    Amount: float = Field(..., ge=-200, le=200, description="Transaction amount (scaled)")
    Time: float = Field(..., ge=-200, le=200, description="Time elapsed (scaled)")


class PredictRequest(BaseModel):
    """Request body for /predict endpoint."""

    features: TransactionFeatures


class PredictResponse(BaseModel):
    """Response body for /predict endpoint."""

    is_fraud: bool
    fraud_probability: float = Field(..., ge=0, le=1)
    risk_level: str = Field(..., pattern="^(LOW|MEDIUM|HIGH)$")


class HealthResponse(BaseModel):
    """Response body for /health endpoint."""

    status: str
    model_version: str
