from fastapi import FastAPI, UploadFile, File, Form, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy.orm import Session
import numpy as np
import cv2
import os
import threading
import tflite_runtime.interpreter as tflite

from database import Base, engine, get_db
import crud
from schemas import PredictionCreate, PredictionOut

# -----------------------------------------------------------
#  INITIAL DATABASE
# -----------------------------------------------------------
Base.metadata.create_all(bind=engine)

app = FastAPI()

# -----------------------------------------------------------
#  CORS (สำคัญมาก)
# -----------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],                # อนุญาตทุก domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)


# -----------------------------------------------------------
#  LOAD TFLITE MODEL
# -----------------------------------------------------------
MODEL_PATH = "model_data/model.tflite"

if not os.path.exists(MODEL_PATH):
    raise Exception("❌ ไม่พบไฟล์ model.tflite — กรุณาอัปโหลดใน backend/model_data")

interpreter = tflite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

model_shape = input_details[0]["shape"]    # (1, 224, 224, 3) หรือ (1, 63)

# ใช้ Lock ป้องกัน crash เวลาเรียกพร้อมกัน
model_lock = threading.Lock()


@app.get("/")
def root():
    return {"message": "Backend Running OK"}


# -----------------------------------------------------------
#  PREDICT (รับรูปภาพ)
# -----------------------------------------------------------
@app.post("/predict")
async def predict(file: UploadFile = File(...), db: Session = Depends(get_db)):

    # ถ้าโมเดลไม่ใช่ input รูป 3 channel
    if len(model_shape) != 4 or model_shape[-1] != 3:
        return {"error": "โมเดลไม่รองรับ input แบบรูปภาพ"}

    contents = await file.read()
    img = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR)

    if img is None:
        return {"error": "อ่านรูปภาพล้มเหลว"}

    # Resize ตามโมเดล
    img = cv2.resize(img, (model_shape[1], model_shape[2]))
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)

    # Predict
    with model_lock:
        interpreter.set_tensor(input_details[0]["index"], img)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]["index"])

    pred_index = int(np.argmax(output))
    confidence = float(np.max(output))
    label = f"class_{pred_index}"

    saved = crud.create_prediction(
        db, PredictionCreate(label=label, confidence=confidence, source="predict")
    )

    return {
        "label": label,
        "confidence": confidence,
        "timestamp": saved.created_at,
    }


# -----------------------------------------------------------
#  TRANSLATE (รับ 63 Landmark)
# -----------------------------------------------------------
class Landmark63(BaseModel):
    points: list[float]  # landmark 63 ค่า


@app.post("/translate")
async def translate_landmarks(payload: Landmark63, db: Session = Depends(get_db)):

    if len(model_shape) != 2:
        return {"error": "โมเดลไม่รองรับ landmark 63 ค่า"}

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
        db, PredictionCreate(label=label, confidence=confidence, source="translate")
    )

    return {
        "label": label,
        "confidence": confidence,
        "timestamp": saved.created_at,
    }


# -----------------------------------------------------------
#  SAVE ONLY (บันทึกลง database)
# -----------------------------------------------------------
@app.post("/save", response_model=PredictionOut)
async def save(data: PredictionCreate, db: Session = Depends(get_db)):
    return crud.create_prediction(db, data)


# -----------------------------------------------------------
#  DATASET ALL
# -----------------------------------------------------------
@app.get("/dataset", response_model=list[PredictionOut])
def get_dataset(db: Session = Depends(get_db)):
    return crud.get_all_predictions(db)


# -----------------------------------------------------------
#  SAVE IMAGE (สำหรับเก็บ dataset จากกล้อง)
# -----------------------------------------------------------
@app.post("/save-image")
async def save_image(
    file: UploadFile = File(...),
    label: str = Form(...),
    db: Session = Depends(get_db),
):
    # เตรียมโฟลเดอร์ dataset/<label>
    save_dir = f"dataset/{label}"
    os.makedirs(save_dir, exist_ok=True)

    # ตั้งชื่อไฟล์
    existing_files = len(os.listdir(save_dir))
    filename = f"img_{existing_files + 1:05d}.jpg"
    filepath = f"{save_dir}/{filename}"

    # บันทึกภาพ
    contents = await file.read()
    with open(filepath, "wb") as f:
        f.write(contents)

    # บันทึกลง database
    saved = crud.create_prediction(
        db,
        PredictionCreate(
            label=label,
            confidence=0.0,
            source="save-image"  # ใช้สำหรับบอกว่าเป็น dataset ที่เก็บเอง
        ),
    )

    return {
        "message": "saved",
        "file": filename,
        "path": filepath,
        "db_id": saved.id,
        "timestamp": saved.created_at,
    }
