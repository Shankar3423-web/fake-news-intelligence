from pydantic import BaseModel, Field

class HealthResponse(BaseModel):
    api_status: str = Field(..., description="Status of the API (e.g., online)")
    model_status: str = Field(..., description="Status of the Model (e.g., loaded)")
    dataset_status: str = Field(..., description="Status of the Dataset (e.g., available)")
