from sqlalchemy import Column, Integer, String, Float, DateTime, Text
from datetime import datetime
from database import Base

class Prediction(Base):
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, index=True)
    label = Column(String, index=True)
    confidence = Column(Float)
    source = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)

class DatasetRecord(Base):
    __tablename__ = "dataset"

    id = Column(Integer, primary_key=True, index=True)
    label = Column(String, index=True)
    sequence = Column(Text)        # เก็บ JSON ของ sequence เฟรม
    timestamp = Column(String)     # string ISO time
