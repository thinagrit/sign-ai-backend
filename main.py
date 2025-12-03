import os
import threading
import numpy as np
import cv2

from fastapi import FastAPI, UploadFile, File, Depends, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy.orm import Session

import tflite_runtime.interpreter as tflite

from database import Base, engine, get_db
import crud
from schemas import PredictionCreate, PredictionOut


# ============================================================
# INIT DATABASE
# ============================================================
Base.metadata.create_all(bind=engine)

app = FastAPI()

# ============================================================
# CORS (สำคัญมากสำหรับ frontend)
# ============================================================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],            # อนุญาตทุก domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================
# LOAD TFLITE MODEL
# ============================================================
interpreter = tflite.Interpreter(model_path="model_data/model.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

model_shape = input_details[0]["shape"]  
# เช่น (1, 224, 224, 3) หรือ (1, 63)

model_lock = threading.Lock()


@app.get("/")
def root():
    return {"message": "Backend Running OK"}


# ============================================================
# PREDICT — ใช้ภาพเข้าโมเดล
# ============================================================
@app.post("/predict")
async def predict(file: UploadFile = File(...), db: Session = Depends(get_db)):

    if model_shape[-1] != 3:
        return {"error": "โมเดลนี้ไม่รองรับภาพ"}

    contents = await file.read()
    img = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR)

    if img is None:
        return {"error": "ไม่สามารถอ่านรูปภาพได้"}

    img = cv2.resize(img, (model_shape[1], model_shape[2]))
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)

    with model_lock:
        interpreter.set_tensor(input_details[0]["index"], img)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]["index"])

    pred_index = int(np.argmax(output))
    confidence = float(np.max(output))
    label = f"class_{pred_index}"

    saved = crud.create_prediction(
        db,
        PredictionCreate(label=label, confidence=confidence, source="predict")
    )

    return {
        "label": label,
        "confidence": confidence,
        "timestamp": saved.created_at,
    }


# ============================================================
# TRANSLATE — รับ landmark 63 ค่าเข้าโมเดล
# ============================================================
class Landmark63(BaseModel):
    points: list[float]  # ความยาวต้อง = model_shape[1]


@app.post("/translate")
async def translate_landmarks(payload: Landmark63, db: Session = Depends(get_db)):

    if model_shape[-1] == 3:
        return {"error": "โมเดลนี้รองรับภาพ ไม่ใช่ landmark"}

    if len(payload.points) != model_shape[1]:
        return {"error": f"ต้องส่ง landmark {model_shape[1]} ค่า"}

    arr = np.array(payload.points, dtype="float32").reshape(1, -1)

    with model_lock:
        interpreter.set_tensor(input_details[0]["index"], arr)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]["index"])

    pred_index = int(np.argmax(output))
    confidence = float(np.max(output))
    label = f"class_{pred_index}"

    saved = crud.create_prediction(
        db,
        PredictionCreate(label=label, confidence=confidence, source="translate")
    )

    return {
        "label": label,
        "confidence": confidence,
        "timestamp": saved.created_at,
    }


# ============================================================
# SAVE DATA (บันทึกข้อความลง DB)
# ============================================================
@app.post("/save", response_model=PredictionOut)
async def save(data: PredictionCreate, db: Session = Depends(get_db)):
    return crud.create_prediction(db, data)


# ============================================================
# GET ALL DATASET
# ============================================================
@app.get("/dataset", response_model=list[PredictionOut])
def get_dataset(db: Session = Depends(get_db)):
    return crud.get_all_predictions(db)


# ============================================================
# SAVE IMAGE FROM CAMERA — สร้าง Dataset โดยภาพ
# ============================================================
@app.post("/save-image")
async def save_image(
    file: UploadFile = File(...),
    label: str = Form(...),
    db: Session = Depends(get_db)
):
    try:
        BASE_DIR = "data"             # โฟลเดอร์หลักเก็บภาพ
        save_dir = f"{BASE_DIR}/{label}"

        # สร้างโฟลเดอร์ถ้ายังไม่มี
        os.makedirs(BASE_DIR, exist_ok=True)
        os.makedirs(save_dir, exist_ok=True)

        # ตรวจสอบว่าที่อยู่เป็นโฟลเดอร์
        if not os.path.isdir(save_dir):
            raise ValueError(f"Path exists but is not a directory: {save_dir}")

        # ตั้งชื่อภาพใหม่
        existing_files = [f for f in os.listdir(save_dir) if f.endswith(".jpg")]
        filename = f"img_{len(existing_files)+1:05d}.jpg"
        filepath = f"{save_dir}/{filename}"

        # บันทึกภาพ
        contents = await file.read()
        with open(filepath, "wb") as f:
            f.write(contents)

        # บันทึกลง DB
        saved = crud.create_prediction(
            db,
            PredictionCreate(label=label, confidence=0.0, source="save-image")
        )

        return {
            "message": "saved",
            "file": filename,
            "path": filepath,
            "db_id": saved.id,
            "timestamp": saved.created_at
        }

    except Exception as e:
        return {"error": str(e)}
