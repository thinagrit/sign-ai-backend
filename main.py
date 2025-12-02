from fastapi import FastAPI, UploadFile, File, Depends, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy.orm import Session
import numpy as np
import cv2
import threading
import os
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
# CORS
# ============================================================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # Allow every frontend domain
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# LOAD MODEL
# ============================================================
MODEL_PATH = "model_data/model.tflite"

if not os.path.exists(MODEL_PATH):
    raise RuntimeError("❌ ERROR: ไม่พบไฟล์ model_data/model.tflite")

interpreter = tflite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# ตัวอย่าง shape:
#   รูปภาพ → (1, 224, 224, 3)
#   Landmark → (1, 63)
model_shape = input_details[0]["shape"]

# Lock กัน crash
model_lock = threading.Lock()


# ============================================================
# ROOT
# ============================================================
@app.get("/")
def root():
    return {"message": "Backend Running OK"}


# ============================================================
# PREDICT (IMAGE INPUT)
# ============================================================
@app.post("/predict")
async def predict(file: UploadFile = File(...), db: Session = Depends(get_db)):

    # ถ้าโมเดลไม่ใช่แบบรูปภาพ
    if model_shape[-1] != 3:
        return {"error": "โมเดลนี้รองรับ landmark เท่านั้น ไม่รองรับรูปภาพ"}

    contents = await file.read()
    img = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR)

    if img is None:
        return {"error": "อ่านรูปภาพไม่ได้"}

    h, w = model_shape[1], model_shape[2]
    img = cv2.resize(img, (w, h))
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
        PredictionCreate(label=label, confidence=confidence, source="predict"),
    )

    return {
        "label": label,
        "confidence": confidence,
        "timestamp": saved.created_at,
    }


# ============================================================
# TRANSLATE (63 LANDMARK VALUES)
# ============================================================
class Landmark63(BaseModel):
    points: list[float]


@app.post("/translate")
async def translate_landmarks(payload: Landmark63, db: Session = Depends(get_db)):

    if model_shape[-1] == 3:
        return {"error": "โมเดลนี้รองรับรูปภาพ ไม่ใช่ landmark"}

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
        PredictionCreate(label=label, confidence=confidence, source="translate"),
    )

    return {
        "label": label,
        "confidence": confidence,
        "timestamp": saved.created_at,
    }


# ============================================================
# SAVE IMAGE (FROM CAMERA)
# ============================================================
@app.post("/save-image")
async def save_image(
    file: UploadFile = File(...),
    label: str = Form(...),
    db: Session = Depends(get_db)
):
    # สร้าง dataset/<label> ถ้ายังไม่มี
    save_dir = f"dataset/{label}"
    os.makedirs(save_dir, exist_ok=True)

    # ชื่อไฟล์ img_00001.jpg
    existing = len(os.listdir(save_dir))
    filename = f"img_{existing + 1:05d}.jpg"
    filepath = f"{save_dir}/{filename}"

    # Save image file
    contents = await file.read()
    with open(filepath, "wb") as f:
        f.write(contents)

    # บันทึก database
    data = PredictionCreate(
        label=label,
        confidence=0.0,
        source="save-image"
    )
    saved = crud.create_prediction(db, data)

    return {
        "message": "saved",
        "path": filepath,
        "db_id": saved.id,
        "timestamp": saved.created_at,
    }


# ============================================================
# GET DATASET (DATABASE)
# ============================================================
@app.get("/dataset", response_model=list[PredictionOut])
def get_dataset(db: Session = Depends(get_db)):
    return crud.get_all_predictions(db)
