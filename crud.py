from sqlalchemy.orm import Session
from models import Prediction

def save_prediction(db: Session, label: str, confidence: float, source: str):
    record = Prediction(
        label=label,
        confidence=confidence,
        source=source
    )
    db.add(record)
    db.commit()
    db.refresh(record)
    return record

def get_history(db: Session):
    return db.query(Prediction).order_by(Prediction.id.desc()).all()
