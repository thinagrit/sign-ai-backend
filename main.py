from fastapi import FastAPI, UploadFile, File, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy.orm import Session
import numpy as np
import cv2
import threading
import tflite_runtime.interpreter as tflite

from database import Base, engine, get_db
import crud
from schemas import PredictionCreate, PredictionOut


# --------------------------
# INIT DB
# --------------------------
Base.metadata.create_all(bind=engine)

app = FastAPI()

# --------------------------
# CORS
# --------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------------------------
# LOAD TFLITE MODEL
# --------------------------
interpreter = tflite.Interpreter(model_path="model_data/model.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Extract model input shape
model_shape = input_details[0]["shape"]   # เช่น (1, 224, 224, 3) หรือ (1, 63)

# Lock ป้องกัน multi-thread crash
model_lock = threading.Lock()


@app.get("/")
def root():
    return {"message": "Backend Running OK"}


# ============================================================
# PREDICT (image input)
# ============================================================
@app.post("/predict")
async def predict(file: UploadFile = File(...), db: Session = Depends(get_db)):

    if model_shape[-1] != 3:
        return {"error": "โมเดลนี้ไม่รองรับรูปภาพ"}

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
# TRANSLATE (63 landmark input)
# ============================================================
class Landmark63(BaseModel):
    points: list[float]   # 63 values


@app.post("/translate")
async def translate_landmarks(payload: Landmark63, db: Session = Depends(get_db)):

    if model_shape[-1] == 3:
        return {"error": "โมเดลนี้รองรับเฉพาะรูปภาพ ไม่ใช่ landmark"}

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
# SAVE (from frontend)
# ============================================================
@app.post("/save", response_model=PredictionOut)
async def save(data: PredictionCreate, db: Session = Depends(get_db)):
    return crud.create_prediction(db, data)


# ============================================================
# GET DATASET
# ============================================================
@app.get("/dataset", response_model=list[PredictionOut])
def get_dataset(db: Session = Depends(get_db)):
    return crud.get_all_predictions(db)
