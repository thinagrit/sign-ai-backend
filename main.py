from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy.orm import Session
import numpy as np
import tflite_runtime.interpreter as tflite

from database import Base, engine, get_db
import crud
from schemas import PredictionCreate, PredictionOut

# -------------------------
# INIT DB
# -------------------------
Base.metadata.create_all(bind=engine)
app = FastAPI()

# -------------------------
# CORS (สำคัญ)
# -------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"msg": "Backend OK - Landmark Model Running"}

# ============================================================
# LOAD MODEL
# ============================================================

import os
import requests
import tflite_runtime.interpreter as tflite

# direct download link ของ model.tflite (RAW file)
MODEL_URL = "https://raw.githubusercontent.com/thinagrit/sign-ai-backend/b25b851a1dff663a766fe048175f3a770ccb4336/model_data/model.tflite"

# ตำแหน่งที่เซฟโมเดลบน Render
MODEL_PATH = "/opt/render/project/src/model.tflite"

# ถ้าโมเดลยังไม่ถูกโหลด -> โหลดจาก URL
if not os.path.exists(MODEL_PATH):
    print("Downloading model from:", MODEL_URL)
    r = requests.get(MODEL_URL)
    with open(MODEL_PATH, "wb") as f:
        f.write(r.content)

# โหลดโมเดล
interpreter = tflite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

INPUT_SIZE = input_details[0]["shape"][1]

# =======================
# LABELS (1–2 ท่า)
# =======================

LABELS = {
    0: "ปวดหัว",    # headache
    1: "จาม",    # sneeze
    # ถ้ามีมากกว่า 2 ก็เพิ่มตรงนี้ได้
}


# ============================================================
# REQUEST MODEL
# ============================================================

class LandmarkInput(BaseModel):
    points: list[float]


@app.post("/translate")
async def translate(payload: LandmarkInput, db: Session = Depends(get_db)):

    # เช็คจำนวน landmark
    if len(payload.points) != INPUT_SIZE:
        return {"error": f"ต้องส่ง {INPUT_SIZE} ค่า แต่ส่งมา {len(payload.points)}"}

    arr = np.array(payload.points, dtype=np.float32).reshape(1, INPUT_SIZE)

    # run model
    interpreter.set_tensor(input_details[0]["index"], arr)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]["index"])

    pred_index = int(np.argmax(output))
    confidence = float(np.max(output))

    label = LABELS.get(pred_index, f"class_{pred_index}")

    # บันทึกลง DB
    saved = crud.create_prediction(
        db,
        PredictionCreate(label=label, confidence=confidence, source="translate")
    )

    return {
        "label": label,
        "confidence": confidence,
        "timestamp": saved.created_at
    }


# ============================================================
# GET DATASET
# ============================================================

@app.get("/dataset", response_model=list[PredictionOut])
def dataset(db: Session = Depends(get_db)):
    return crud.get_all_predictions(db)
