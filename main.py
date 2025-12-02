from fastapi import FastAPI, UploadFile, File, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy.orm import Session
import numpy as np
import cv2
import tflite_runtime.interpreter as tflite

from database import Base, engine, get_db
import crud
from schemas import PredictionCreate, PredictionOut


# --------------------------
#  INIT DB
# --------------------------
Base.metadata.create_all(bind=engine)

app = FastAPI()

# --------------------------
#  CORS
# --------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],      # ให้ frontend call ได้ทุกโดเมน
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------------------------
#  LOAD TFLITE MODEL
# --------------------------
interpreter = tflite.Interpreter(model_path="model_data/model.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()


@app.get("/")
def root():
    return {"message": "Backend Running OK"}


# ============================================================
#  PREDICT (อัปโหลดรูปภาพ)
# ============================================================
@app.post("/predict")
async def predict(file: UploadFile = File(...), db: Session = Depends(get_db)):
    contents = await file.read()
    img = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR)

    # เตรียมภาพ
    img = cv2.resize(img, (224, 224))
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)

    # ส่งเข้าโมเดล
    interpreter.set_tensor(input_details[0]["index"], img)
    interpreter.invoke()

    output = interpreter.get_tensor(output_details[0]["index"])
    pred_index = int(np.argmax(output))
    confidence = float(np.max(output))

    label = f"class_{pred_index}"

    # บันทึกลง database
    data = PredictionCreate(label=label, confidence=confidence, source="predict")
    saved = crud.create_prediction(db, data)

    return {
        "label": label,
        "confidence": confidence,
        "timestamp": saved.created_at,
    }


# ============================================================
#  TRANSLATE (รับ landmark 63 ค่า)
# ============================================================

# รับ landmarks จาก frontend
class Landmark63(BaseModel):
    points: list[float]   # 63 ค่า


@app.post("/translate")
async def translate_landmarks(payload: Landmark63, db: Session = Depends(get_db)):
    points = np.array(payload.points, dtype="float32")

    if len(points) != 63:
        return {"error": "landmark ต้องมี 63 ค่า"}

    # แปลงเป็น input shape = (1, 63)
    arr = points.reshape(1, -1)

    interpreter.set_tensor(input_details[0]["index"], arr)
    interpreter.invoke()

    output = interpreter.get_tensor(output_details[0]["index"])
    pred_index = int(np.argmax(output))
    confidence = float(np.max(output))

    label = f"class_{pred_index}"

    # บันทึกลง database
    data = PredictionCreate(label=label, confidence=confidence, source="translate")
    saved = crud.create_prediction(db, data)

    return {
        "label": label,
        "confidence": confidence,
        "timestamp": saved.created_at,
    }


# ============================================================
#  SAVE จาก frontend (Predict หรือ Translate)
# ============================================================
@app.post("/save", response_model=PredictionOut)
async def save(data: PredictionCreate, db: Session = Depends(get_db)):
    return crud.create_prediction(db, data)


# ============================================================
#  ดึง Dataset ทั้งหมด
# ============================================================
@app.get("/dataset", response_model=list[PredictionOut])
def get_dataset(db: Session = Depends(get_db)):
    return crud.get_all_predictions(db)
