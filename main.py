# backend/main.py
from fastapi import FastAPI, Depends, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from pydantic import BaseModel
import numpy as np
import os, json
from datetime import datetime

from database import engine, SessionLocal
import models, schemas, crud

from tensorflow import keras

# -------------------------------------------
# 🔹 สร้างตาราง DB ถ้ายังไม่มี
# -------------------------------------------
models.Base.metadata.create_all(bind=engine)

app = FastAPI(title="Sign-AI-Med Backend")

# -------------------------------------------
# 🔹 อนุญาตให้ frontend เรียกได้
# -------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # ใส่ URL Vercel จริงได้
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------------------
# 🔹 DB Session Dependency
# -------------------------------------------
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# -------------------------------------------
# 🔹 โหลดโมเดลแบบ 63 ค่า
# -------------------------------------------

MODEL_PATH = "model_data/model.h5"
LABEL_PATH = "model_data/labels.txt"
os.makedirs("model_data", exist_ok=True)

def create_dummy_model():
    """สร้างโมเดล dummy (input = 63 ค่า)"""
    print("⚙️ Creating dummy model...")

    model = keras.Sequential([
        keras.layers.Input(shape=(63,)),
        keras.layers.Dense(64, activation="relu"),
        keras.layers.Dense(32, activation="relu"),
        keras.layers.Dense(5, activation="softmax"),
    ])
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    X = np.random.rand(50, 63)
    y = np.random.randint(0, 5, size=(50,))
    model.fit(X, y, epochs=1, verbose=0)

    model.save(MODEL_PATH)

    with open(LABEL_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(["Hello", "Thanks", "ILoveYou", "Yes", "No"]))

    print("✅ Dummy model created.")

# โหลดโมเดลจริงหรือสร้าง dummy
try:
    if not os.path.exists(MODEL_PATH) or not os.path.exists(LABEL_PATH):
        create_dummy_model()

    model = keras.models.load_model(MODEL_PATH)
    with open(LABEL_PATH, "r", encoding="utf-8") as f:
        labels = [line.strip() for line in f.readlines()]

    use_tf = True
    print("✅ TensorFlow model loaded successfully.")

except Exception as e:
    use_tf = False
    print("❌ Model load error:", e)


# ===========================================
#               ROUTES API
# ===========================================

@app.get("/")
def index():
    return {"status": "ok", "message": "Sign-AI-Med backend is running"}


# -------------------------------------------
# 🔹 /predict → ทำนายจาก landmark 63 ค่า
# -------------------------------------------
class LandmarkInput(BaseModel):
    values: list[float]  # ต้องมี 63 ค่า


@app.post("/predict", response_model=schemas.PredictionOut)
def predict_landmarks(data: LandmarkInput, db: Session = Depends(get_db)):

    if not use_tf:
        raise HTTPException(status_code=500, detail="Model not loaded")

    if len(data.values) != 63:
        raise HTTPException(status_code=400, detail="Input must be 63 float values")

    vec = np.array(data.values).reshape(1, -1)

    try:
        pred = model.predict(vec)
        class_id = int(np.argmax(pred, axis=1)[0])
        confidence = float(np.max(pred, axis=1)[0])
        label = labels[class_id] if class_id < len(labels) else str(class_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {e}")

    pred_in = schemas.PredictionCreate(
        label=label,
        confidence=confidence,
        source="web-landmark",
        meta=None,
    )
    saved = crud.create_prediction(db=db, pred=pred_in)

    return saved


# -------------------------------------------
# 🔹 /history → ดูประวัติทั้งหมด
# -------------------------------------------
@app.get("/history", response_model=list[schemas.PredictionOut])
def get_history(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, le=1000),
    db: Session = Depends(get_db)
):
    return crud.get_predictions(db=db, skip=skip, limit=limit)


# -------------------------------------------
# 🔹 /api/v1/collect → บันทึก dataset
# -------------------------------------------
class SequenceCollect(BaseModel):
    label: str
    sequence: list          # list ของ landmark
    timestamp: str


@app.post("/api/v1/collect")
def collect_sequence(data: SequenceCollect):
    os.makedirs("collected_data", exist_ok=True)

    safe_label = data.label.replace(" ", "_").lower()
    timestamp = data.timestamp.replace(":", "-")

    label_dir = os.path.join("collected_data", safe_label)
    os.makedirs(label_dir, exist_ok=True)

    file_path = os.path.join(label_dir, f"{timestamp}.json")

    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data.dict(), f, ensure_ascii=False, indent=2)

    return {"status": "ok", "saved_to": file_path}


# -------------------------------------------
# 🔹 /api/v1/infer/sequence → แบบ sequence input
# -------------------------------------------
class SequencePayload(BaseModel):
    device_id: str
    timestamp: int
    sequence: list[list[float]]    # SEQ × 63


@app.post("/api/v1/infer/sequence", response_model=schemas.PredictionOut)
def infer_sequence(payload: SequencePayload, db: Session = Depends(get_db)):

    seq = np.array(payload.sequence).flatten().reshape(1, -1)

    pred = model.predict(seq)
    class_id = int(np.argmax(pred, axis=1)[0])
    confidence = float(np.max(pred, axis=1)[0])
    label = labels[class_id] if class_id < len(labels) else str(class_id)

    pred_in = schemas.PredictionCreate(
        label=label,
        confidence=confidence,
        source=payload.device_id,
        meta=None,
    )
    saved = crud.create_prediction(db, pred_in)

    return saved


# -------------------------------------------
# 🔹 /api/v1/dataset → list dataset
# -------------------------------------------
@app.get("/api/v1/dataset")
def list_dataset():
    collected_dir = "collected_data"
    os.makedirs(collected_dir, exist_ok=True)

    files = []
    for root, _, filenames in os.walk(collected_dir):
        for f in filenames:
            if f.endswith(".json"):
                files.append(os.path.relpath(os.path.join(root, f), collected_dir))

    return {"count": len(files), "files": files}
