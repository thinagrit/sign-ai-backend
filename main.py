from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy.orm import Session
import numpy as np
import os
import requests
import tflite_runtime.interpreter as tflite

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
# MODEL DOWNLOAD (GitHub Release Asset)
# ============================================================

MODEL_URL = (
    "https://github.com/thinagrit/sign-ai-backend/releases/"
    "download/v1.0.0/model.tflite"
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
# LOAD TFLITE (tflite-runtime)
# ============================================================

interpreter = tflite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

INPUT_SIZE = input_details[0]["shape"][1]

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
# CORE INFERENCE FUNCTION
# ============================================================

def run_inference(points: list[float]):
    if len(points) != INPUT_SIZE:
        raise HTTPException(
            status_code=422,
            detail=f"ต้องส่ง {INPUT_SIZE} ค่า แต่ส่งมา {len(points)}"
        )

    arr = np.array(points, dtype=np.float32).reshape(1, INPUT_SIZE)
    interpreter.set_tensor(input_details[0]["index"], arr)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]["index"])

    idx = int(np.argmax(output))
    conf = float(np.max(output))
    label = LABELS.get(idx, "unknown")

    return label, conf

# ============================================================
# API ROUTES
# ============================================================

@app.post("/translate")
@app.post("/predict")  # 🔥 รองรับ frontend เดิม
def predict(payload: LandmarkInput, db: Session = Depends(get_db)):

    label, confidence = run_inference(payload.points)

    saved = crud.create_prediction(
        db,
        PredictionCreate(
            label=label,
            confidence=confidence,
            source="api"
        )
    )

    return {
        "label": label,
        "confidence": confidence,
        "timestamp": saved.created_at
    }

@app.get("/dataset", response_model=list[PredictionOut])
def dataset(db: Session = Depends(get_db)):
    return crud.get_all_predictions(db)
predictions(db)
