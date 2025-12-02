from sqlalchemy.orm import Session
import models, schemas

def create_prediction(db: Session, pred: schemas.PredictionCreate):
    db_item = models.Prediction(
        label=pred.label,
        confidence=pred.confidence,
        source=pred.source,
    )
    db.add(db_item)
    db.commit()
    db.refresh(db_item)
    return db_item

def get_predictions(db: Session, skip: int = 0, limit: int = 100):
    return db.query(models.Prediction).offset(skip).limit(limit).all()
