from pydantic import BaseModel
from typing import Optional

class TransactionRequest(BaseModel):
    '''
    This defines the exact JSON structure and datatypes expected by the API.
    '''

    # Temporal data
    TransactionDT: int

    # Aggregation Anchors
    card1: int
    TransactionAmt: float

    # categorical data
    P_emaildomain: Optional[str] = 'unknown'