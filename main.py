import json
from typing import List, Optional

import numpy as np
import tensorflow as tf
from fastapi import FastAPI, UploadFile, File, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from database import SessionLocal, engine
import models
import crud

# สร้างตารางใน DB ถ้ายังไม่มี
models.Base.metadata.create_all(bind=engine)

app = FastAPI()

# CORS ให้ frontend เรียกได้
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # ถ้ารู้โดเมนจริง ค่อยเปลี่ยนจาก "*" เป็นโดเมน
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# โหลด labels
with open("model_data/labels.txt", "r", encoding="utf-8") as f:
    LABELS = f.read().splitlines()

# โหลด TFLite model ด้วย TensorFlow
interpreter = tf.lite.Interpreter(model_path="model_data/model.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

class DatasetPayload(BaseModel):
    label: str
    sequence: List[List[float]]   # list ของเฟรม แต่ละเฟรม = 63 ค่า
    timestamp: str


@app.get("/")
def root():
    return {"message": "Backend is running (TFLite via TF 2.20)", "labels": LABELS}


@app.post("/predict")
async def predict(file: UploadFile = File(...), db=Depends(get_db)):
    """
    รับไฟล์ JSON ในรูปแบบ:
    [0.1, -0.2, ..., (63 ค่า)]
    """
    raw = await file.read()

    try:
        arr = json.loads(raw.decode("utf-8"))
        arr = np.array(arr, dtype=np.float32)
    except Exception:
        return {"error": "รูปแบบไฟล์ไม่ถูกต้อง ต้องเป็น JSON array ของตัวเลข 63 ค่า"}

    if arr.shape[0] != 63:
        return {"error": f"ต้องมี 63 ค่า แต่ส่งมา {arr.shape[0]} ค่า"}

    # reshape สำหรับโมเดล
    x = np.expand_dims(arr, axis=0)

    # ใส่เข้า TFLite
    interpreter.set_tensor(input_details[0]["index"], x)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]["index"])[0]

    class_id = int(np.argmax(output))
    confidence = float(np.max(output))
    label = LABELS[class_id]

    crud.save_prediction(db, label=label, confidence=confidence, source="frontend")

    return {
        "label": label,
        "confidence": round(confidence, 4),
        "class_id": class_id,
    }


@app.post("/api/v1/collect")
def collect_dataset(payload: DatasetPayload, db=Depends(get_db)):
    """
    รับ sequence landmark สำหรับเก็บ dataset
    payload.sequence = list ของเฟรมๆ ละ 63 ค่า
    """
    crud.save_dataset_record(
        db,
        label=payload.label,
        sequence=payload.sequence,
        timestamp=payload.timestamp,
    )
    return {"message": "dataset saved"}


@app.get("/history")
def history(limit: int = 50, db=Depends(get_db)):
    rows = crud.get_history(db, limit=limit)
    return [
        {
            "id": r.id,
            "label": r.label,
            "confidence": r.confidence,
            "source": r.source,
            "created_at": r.created_at,
        }
        for r in rows
    ]
