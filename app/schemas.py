from pydantic import BaseModel, Field


class LoanApplication(BaseModel):
    total_investment: float = Field(
        ...,
        description="Total investment amount",
    )
    current_balance: float = Field(
        ...,
        description="Current account balance",
    )
    marital_status: str = Field(
        ...,
        description="Customer marital status",
    )
    gender: str = Field(
        ...,
        description="Customer gender",
    )
    due_payment: float = Field(
        ...,
        description="Amount of payment currently due",
    )
    compensation_charged: str = Field(
        ...,
        description="Compensation charged category",
    )
    client_type: str = Field(
        ...,
        description="Customer/client type",
    )
    repay_mode: str = Field(
        ...,
        description="Loan repayment mode",
    )


class PredictionResponse(BaseModel):
    prediction: str
    probability_good: float
    probability_bad: float