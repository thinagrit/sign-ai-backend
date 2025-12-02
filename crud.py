from sqlalchemy.orm import Session
from models import Prediction
from schemas import PredictionCreate


def create_prediction(db: Session, data: PredictionCreate):
    """
    บันทึกข้อมูลการทำนาย (Predict / Translate)
    """
    item = Prediction(
        label=data.label,
        confidence=data.confidence,
        source=data.source,
    )
    db.add(item)

    try:
        db.commit()
        db.refresh(item)
    except Exception as e:
        db.rollback()
        raise e

    return item


def get_all_predictions(db: Session):
    """
    ดึงข้อมูลทั้งหมดจากฐานข้อมูล โดยเรียงจากใหม่ → เก่า
    """
    return db.query(Prediction).order_by(Prediction.id.desc()).all()
