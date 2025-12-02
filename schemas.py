from pydantic import BaseModel
from datetime import datetime

class PredictionBase(BaseModel):
    label: str
    confidence: float
    source: str
    meta: dict | None = None

class PredictionCreate(PredictionBase):
    pass

class PredictionOut(PredictionBase):
    id: int
    created_at: datetime

    class Config:
        orm_mode = True
