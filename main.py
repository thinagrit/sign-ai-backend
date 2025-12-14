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

# -------------------------
# INIT APP & DB
# -------------------------
Base.metadata.create_all(bind=engine)
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"status": "OK", "model": "Sign AI (2 gestures)"}

# -------------------------
# MODEL CONFIG
# -------------------------
MODEL_URL = "https://github.com/thinagrit/sign-ai-backend/releases/download/v1.0.0/model.tflite"
MODEL_PATH = "model.tflite"

if not os.path.exists(MODEL_PATH):
    print("⬇️ Downloading model...")
    r = requests.get(MODEL_URL)
    r.raise_for_status()   # <-- ถ้า 404 จะ error ตรงนี้
    with open(MODEL_PATH, "wb") as f:
        f.write(r.content)

interpreter = tflite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

INPUT_SIZE = input_details[0]["shape"][1]

LABELS = {
    0: "ปวดหัว",
    1: "จาม",
}

# -------------------------
# API
# -------------------------
class LandmarkInput(BaseModel):
    points: list[float]

@app.post("/translate")
def translate(payload: LandmarkInput, db: Session = Depends(get_db)):
    if len(payload.points) != INPUT_SIZE:
        return {"error": f"ต้องส่ง {INPUT_SIZE} ค่า"}

    arr = np.array(payload.points, dtype=np.float32).reshape(1, INPUT_SIZE)

    interpreter.set_tensor(input_details[0]["index"], arr)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]["index"])

    idx = int(np.argmax(output))
    conf = float(np.max(output))
    label = LABELS.get(idx, "unknown")

    saved = crud.create_prediction(
        db,
        PredictionCreate(label=label, confidence=conf, source="api")
    )

    return {
        "label": label,
        "confidence": conf,
        "timestamp": saved.created_at
    }

@app.get("/dataset", response_model=list[PredictionOut])
def dataset(db: Session = Depends(get_db)):
    return crud.get_all_predictions(db)
