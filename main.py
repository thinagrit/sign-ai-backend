from fastapi import FastAPI, Depends
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
# INIT APP + DB
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
    return {"status": "OK", "model": "TFLite", "classes": ["ปวดหัว", "จาม"]}

# ============================================================
# LOAD MODEL FROM GITHUB RELEASES
# ============================================================

MODEL_URL = "https://github.com/thinagrit/sign-ai-backend/releases/download/v1.0.0/model.tflite"
MODEL_PATH = "model.tflite"

if not os.path.exists(MODEL_PATH):
    print("📥 Downloading model from GitHub Releases...")
    r = requests.get(MODEL_URL, timeout=60)
    r.raise_for_status()
    with open(MODEL_PATH, "wb") as f:
        f.write(r.content)

print("✅ Model ready")

interpreter = tflite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

INPUT_SIZE = input_details[0]["shape"][1]

# ============================================================
# LABELS
# ============================================================

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
# TRANSLATE ENDPOINT
# ============================================================

@app.post("/translate")
def translate(payload: LandmarkInput, db: Session = Depends(get_db)):

    if len(payload.points) != INPUT_SIZE:
        return {
            "error": f"ต้องส่ง {INPUT_SIZE} ค่า แต่ส่งมา {len(payload.points)}"
        }

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
            source="translate"
        )
    )

    return {
        "label": label,
        "confidence": round(confidence, 4),
        "timestamp": saved.created_at
    }

# ============================================================
# HISTORY ENDPOINT
# ============================================================

@app.get("/dataset", response_model=list[PredictionOut])
def dataset(db: Session = Depends(get_db)):
    return crud.get_all_predictions(db)
