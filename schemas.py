from pydantic import BaseModel, Field
from datetime import datetime


# -----------------------------------------------------------------
# Base Schema
# -----------------------------------------------------------------
class PredictionBase(BaseModel):
    label: str = Field(..., description="ผลลัพธ์จากโมเดล เช่น class_0")
    confidence: float = Field(..., ge=0, le=1, description="ความมั่นใจของโมเดล 0 - 1")
    source: str = Field(..., description="predict หรือ translate")


# -----------------------------------------------------------------
# สำหรับสร้างข้อมูลใหม่
# -----------------------------------------------------------------
class PredictionCreate(PredictionBase):
    """
    ใช้ตอนบันทึกข้อมูลเข้า Database
    """
    pass


# -----------------------------------------------------------------
# สำหรับตอบกลับ frontend
# -----------------------------------------------------------------
class PredictionOut(PredictionBase):
    id: int
    created_at: datetime

    class Config:
        orm_mode = True   # ใช้กับ SQLAlchemy
