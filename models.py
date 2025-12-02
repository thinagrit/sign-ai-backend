from sqlalchemy import Column, Integer, String, Float, DateTime
from datetime import datetime
from database import Base


class Prediction(Base):
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, index=True)

    # ชื่อ label เช่น class_0
    label = Column(String(255), nullable=False, index=True)

    # ความมั่นใจ 0.0 - 1.0
    confidence = Column(Float, nullable=False)

    # มาจาก predict หรือ translate
    source = Column(String(50), nullable=False)

    # วันเวลาบันทึก
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    def __repr__(self):
        return f"<Prediction id={self.id} label={self.label} confidence={self.confidence}>"
