from pydantic import BaseModel, Field
from typing import Optional

class TransactionInput(BaseModel):
    """
    The strict Pydantic data contract for incoming live transaction payloads.
    Enforces explicit type validation at the gateway boundary to catch corrupt,
    malformed, or down-typed data parameters before feature engineering.
    """
    TransactionID: int = Field(..., description="Unique identifier for the transaction record")
    TransactionDT: int = Field(..., description="Timedelta from a given reference datetime in seconds")
    TransactionAmt: float = Field(..., description="Transaction payment amount in USD")
    ProductCD: str = Field(..., description="Product code for the transaction item category")
    card1: int = Field(..., description="Payment card identifier feature 1")
    card2: Optional[float] = Field(None, description="Payment card identifier feature 2")
    card3: Optional[float] = Field(None, description="Payment card identifier feature 3")
    card4: Optional[str] = Field(None, description="Card issuing network provider name")
    card5: Optional[float] = Field(None, description="Payment card identifier feature 5")
    card6: Optional[str] = Field(None, description="Payment card funding category type")
    P_emaildomain: Optional[str] = Field(None, description="Purchaser contact email domain address")
    R_emaildomain: Optional[str] = Field(None, description="Recipient contact email domain address")

    # Core Counting features used by the model
    C1: Optional[float] = Field(0.0, description="Counting feature 1")
    C2: Optional[float] = Field(0.0, description="Counting feature 2")
    C11: Optional[float] = Field(0.0, description="Counting feature 11")
    C12: Optional[float] = Field(0.0, description="Counting feature 12")
    C14: Optional[float] = Field(0.0, description="Counting feature 14")

    # Primary behavioral validation tracking blocks
    V70: Optional[float] = Field(0.0, description="Vesta behavioral match count feature 70")
    V91: Optional[float] = Field(0.0, description="Vesta behavioral match count feature 91")
    V147: Optional[float] = Field(0.0, description="Vesta behavioral match count feature 147")
    V156: Optional[float] = Field(0.0, description="Vesta behavioral match count feature 156")
    V172: Optional[float] = Field(0.0, description="Vesta behavioral match count feature 172")
    V258: Optional[float] = Field(0.0, description="Vesta behavioral match count feature 258")
    V294: Optional[float] = Field(0.0, description="Vesta behavioral match count feature 294")

    # Metadata fields for identity grouping matches
    id_17: Optional[float] = Field(None, description="Identity proxy proxy feature 17")
    DeviceType: Optional[str] = Field(None, description="Operating system platform architecture class name")
    DeviceInfo: Optional[str] = Field(None, description="Device browser build identification string")

    model_config = {
        "json_schema_extra": {
            "example": {
                "TransactionID": 3663549,
                "TransactionDT": 86400,
                "TransactionAmt": 150.75,
                "ProductCD": "W",
                "card1": 13926,
                "card2": 555.0,
                "card3": 150.0,
                "card4": "discover",
                "card5": 142.0,
                "card6": "credit",
                "P_emaildomain": "gmail.com",
                "R_emaildomain": "gmail.com",
                "C1": 1.0,
                "C2": 1.0,
                "C11": 1.0,
                "C12": 0.0,
                "C14": 1.0,
                "V70": 1.0,
                "V91": 0.0,
                "V147": 0.0,
                "V156": 0.0,
                "V172": 0.0,
                "V258": 1.0,
                "V294": 0.0,
                "id_17": 166.0,
                "DeviceType": "desktop",
                "DeviceInfo": "Windows"
            }
        }
    }

class FraudPredictionResponse(BaseModel):
    """
    The structured output contract returned to upstream consuming apps.
    Guarantees consistent, safe text structures for downstream accounting parsers.
    """
    transaction_id: int
    is_fraud: int
    fraud_probability: float
    decision_threshold: float
    status: str