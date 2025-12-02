from sqlalchemy.orm import Session
from models import Prediction, DatasetRecord
import json

def save_prediction(db: Session, label: str, confidence: float, source: str):
    record = Prediction(
        label=label,
        confidence=confidence,
        source=source,
    )
    db.add(record)
    db.commit()
    db.refresh(record)
    return record

def get_history(db: Session, limit: int = 50):
    return (
        db.query(Prediction)
        .order_by(Prediction.id.desc())
        .limit(limit)
        .all()
    )

def save_dataset_record(db: Session, label: str, sequence, timestamp: str):
    record = DatasetRecord(
        label=label,
        sequence=json.dumps(sequence),
        timestamp=timestamp,
    )
    db.add(record)
    db.commit()
    db.refresh(record)
    return record

def get_dataset(db: Session, label: str | None = None):
    q = db.query(DatasetRecord)
    if label:
        q = q.filter(DatasetRecord.label == label)
    return q.order_by(DatasetRecord.id.desc()).all()
