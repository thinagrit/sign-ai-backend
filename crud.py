from sqlalchemy.orm import Session
from models import Prediction
from schemas import PredictionCreate

def create_prediction(db: Session, data: PredictionCreate):
    item = Prediction(
        label=data.label,
        confidence=data.confidence,
        source=data.source,
    )
    db.add(item)
    db.commit()
    db.refresh(item)
    return item


def get_all_predictions(db: Session):
    return db.query(Prediction).order_by(Prediction.id.desc()).all()
