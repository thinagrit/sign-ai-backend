from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy.orm import Session
import numpy as np
import os
import tensorflow as tf

from database import Base, engine, get_db
import crud
from schemas import PredictionCreate, PredictionOut

# ============================================================
# INIT DB & APP
# ============================================================

Base.metadata.create_all(bind=engine)

app = FastAPI(title="Sign AI Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # ใช้กับ Vercel
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"status": "OK", "model": "Sign AI (headache, sneeze)"}

# ============================================================
# LOAD MODEL (LOCAL FILE – แนะนำที่สุด)
# ============================================================

MODEL_PATH = "model_data/model.tflite"

if not os.path.exists(MODEL_PATH):
    raise RuntimeError("❌ model.tflite not found in model_data/")

interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

INPUT_SIZE = int(input_details[0]["shape"][1])

LABELS = {
    0: "ปวดหัว",
    1: "จาม",
}

# ============================================================
# REQUEST SCHEMA
# ============================================================

class LandmarkInput(BaseModel):
    points: list[float]

# ============================================================
# API: /predict  (🔥 frontend เรียก path นี้)
# ============================================================

@app.post("/predict")
def predict(payload: LandmarkInput, db: Session = Depends(get_db)):

    if len(payload.points) != INPUT_SIZE:
        raise HTTPException(
            status_code=400,
            detail=f"ต้องส่ง {INPUT_SIZE} ค่า แต่ส่งมา {len(payload.points)}"
        )

    x = np.array(payload.points, dtype=np.float32).reshape(1, INPUT_SIZE)

    interpreter.set_tensor(input_details[0]["index"], x)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]["index"])

    pred_index = int(np.argmax(output))
    confidence = float(np.max(output))
    label = LABELS.get(pred_index, "unknown")

    saved = crud.create_prediction(
        db,
        PredictionCreate(
            label=label,
            confidence=confidence,
            source="predict"
        )
    )

    return {
        "label": label,
        "confidence": confidence,
        "timestamp": saved.created_at
    }

# ============================================================
# API: dataset (optional)
# ============================================================

@app.get("/dataset", response_model=list[PredictionOut])
def dataset(db: Session = Depends(get_db)):
    return crud.get_all_predictions(db)
