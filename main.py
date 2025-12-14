from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy.orm import Session
import numpy as np

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
# LOAD MODEL (โหลดจาก GitHub Releases)
# ============================================================

import os
import requests
import tflite_runtime.interpreter as tflite

# ⭐ ลิงก์ดาวน์โหลดโมเดลจาก GitHub Releases (binary ไฟล์จริง)
MODEL_URL = "https://github.com/thinagrit/sign-ai-backend/releases/download/v1.0.0/model.tflite"

# ตำแหน่งเก็บโมเดลบน Render
MODEL_PATH = "/opt/render/project/src/model.tflite"

# 📌 ดาวน์โหลดโมเดลถ้ายังไม่มีในระบบ
if not os.path.exists(MODEL_PATH):
    print("Downloading model from:", MODEL_URL)
    r = requests.get(MODEL_URL)
    with open(MODEL_PATH, "wb") as f:
        f.write(r.content)

# 📌 โหลด TFLite Interpreter
interpreter = tflite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

# ดึงข้อมูล input/output
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
INPUT_SIZE = input_details[0]["shape"][1]  # จำนวน landmark (เช่น 63 ค่า)

# =======================
# LABELS (ปรับตามจำนวนคลาส)
# =======================

LABELS = {
    0: "ปวดหัว",     # headache
    1: "จาม",         # sneeze
}

# ============================================================
# REQUEST MODEL
# ============================================================

class LandmarkInput(BaseModel):
    points: list[float]


@app.post("/translate")
async def translate(payload: LandmarkInput, db: Session = Depends(get_db)):

    # ตรวจสอบจำนวน landmark
    if len(payload.points) != INPUT_SIZE:
        return {"error": f"ต้องส่ง {INPUT_SIZE} ค่า แต่ส่งมา {len(payload.points)}"}

    arr = np.array(payload.points, dtype=np.float32).reshape(1, INPUT_SIZE)

    # รันโมเดล
    interpreter.set_tensor(input_details[0]["index"], arr)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]["index"])

    pred_index = int(np.argmax(output))
    confidence = float(np.max(output))
    label = LABELS.get(pred_index, f"class_{pred_index}")

    # บันทึกผลลงฐานข้อมูล
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
