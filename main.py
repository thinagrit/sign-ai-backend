from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy.orm import Session
import numpy as np
import os
import requests
import tensorflow as tf

from database import Base, engine, get_db
import crud
from schemas import PredictionCreate, PredictionOut

# ============================================================
# INIT APP & DB
# ============================================================

Base.metadata.create_all(bind=engine)

app = FastAPI(title="Sign AI Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"status": "OK", "model": "Sign AI (headache, sneeze)"}

# ============================================================
# MODEL DOWNLOAD (GitHub Releases)
# ============================================================

MODEL_URL = (
    "https://github.com/thinagrit/sign-ai-backend/"
    "releases/download/v1.0.0/model.tflite"
)
MODEL_PATH = "model.tflite"

if not os.path.exists(MODEL_PATH):
    print("⬇️ Downloading model...")
    r = requests.get(MODEL_URL, timeout=60)
    r.raise_for_status()
    with open(MODEL_PATH, "wb") as f:
        f.write(r.content)
    print("✅ Model downloaded")

# ============================================================
# LOAD TFLITE
# ============================================================

interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

INPUT_SIZE = input_details[0]["shape"][1]

LABELS = {
    0: "ปวดหัว",
    1: "จาม",
}

# ============================================================
# REQUEST SCHEMA (Flexible)
# ============================================================

class PredictInput(BaseModel):
    points: list[float] | None = None
    landmarks: list[float] | None = None
    data: list[float] | None = None

# ============================================================
# CORE PREDICT FUNCTION
# ============================================================

def run_model(values: list[float]):
    if len(values) != INPUT_SIZE:
        raise HTTPException(
            status_code=422,
            detail=f"ต้องส่ง {INPUT_SIZE} ค่า แต่ส่งมา {len(values)}"
        )

    arr = np.array(values, dtype=np.float32).reshape(1, INPUT_SIZE)

    interpreter.set_tensor(input_details[0]["index"], arr)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]["index"])

    idx = int(np.argmax(output))
    confidence = float(np.max(output))
    label = LABELS.get(idx, "unknown")

    return label, confidence

# ============================================================
# API: /predict (Frontend ใช้)
# ============================================================

@app.post("/predict")
def predict(payload: PredictInput, db: Session = Depends(get_db)):

    values = (
        payload.points
        or payload.landmarks
        or payload.data
    )

    if values is None:
        raise HTTPException(
            status_code=422,
            detail="ต้องส่ง points หรือ landmarks หรือ data"
        )

    label, confidence = run_model(values)

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
# API: /translate (ของเดิม)
# ============================================================

@app.post("/translate")
def translate(payload: PredictInput, db: Session = Depends(get_db)):

    values = (
        payload.points
        or payload.landmarks
        or payload.data
    )

    if values is None:
        raise HTTPException(
            status_code=422,
            detail="ต้องส่ง points หรือ landmarks หรือ data"
        )

    label, confidence = run_model(values)

    saved = crud.create_prediction(
        db,
        PredictionCreate(
            label=label,
            confidence=confidence,
            source="translate"
        )
    )

    return {
        "label": label,
        "confidence": confidence,
        "timestamp": saved.created_at
    }

# ============================================================
# DATASET
# ============================================================

@app.get("/dataset", response_model=list[PredictionOut])
def dataset(db: Session = Depends(get_db)):
    return crud.get_all_predictions(db)
