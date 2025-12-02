import numpy as np
from fastapi import FastAPI, UploadFile, File, Depends
import tflite_runtime.interpreter as tflite
from database import SessionLocal, engine
from fastapi.middleware.cors import CORSMiddleware
import crud
import models
import json
import uvicorn

# สร้างตาราง DB ถ้ายังไม่มี
models.Base.metadata.create_all(bind=engine)

app = FastAPI()

# อนุญาตให้ frontend (Vercel) เรียกได้
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # ถ้ารู้โดเมนจริงแนะนำใส่แทน "*"
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# โหลด labels
with open("model_data/labels.txt", "r", encoding="utf-8") as f:
    LABELS = f.read().splitlines()

# โหลดโมเดล TFLite
interpreter = tflite.Interpreter(model_path="model_data/model.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Dependency สำหรับ DB session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@app.get("/")
def root():
    return {"message": "Backend is running (TFLite)", "classes": LABELS}


@app.post("/predict")
async def predict(file: UploadFile = File(...), db=Depends(get_db)):
    """
    รับไฟล์ JSON หรือ TXT ที่มี landmark 63 ค่า
    เช่น: [0.1, 0.4, 0.23, ...]
    """

    raw = await file.read()
    try:
        # assume data เป็น JSON array 63 ค่า
        data = json.loads(raw.decode("utf-8"))
        arr = np.array(data).astype("float32")
    except:
        return {"error": "รูปแบบข้อมูลไม่ถูกต้อง ต้องส่ง JSON array ยาว 63 ค่า"}

    if arr.shape[0] != 63:
        return {"error": "input ต้องยาว 63 ค่า"}

    # reshape สำหรับโมเดล
    arr = np.expand_dims(arr, axis=0)

    # ใส่ข้อมูลเข้าโมเดล TFLite
    interpreter.set_tensor(input_details[0]['index'], arr)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])

    # Prediction
    class_id = int(np.argmax(output))
    confidence = float(np.max(output))
    label = LABELS[class_id]

    # บันทึก DB
    crud.save_prediction(
        db=db,
        label=label,
        confidence=confidence,
        source="landmark-63"
    )

    return {
        "label": label,
        "class_id": class_id,
        "confidence": round(confidence, 4)
    }


@app.get("/history")
def history(db=Depends(get_db)):
    result = crud.get_history(db)
    formatted = [
        {
            "id": r.id,
            "label": r.label,
            "confidence": r.confidence,
            "source": r.source,
            "created_at": r.created_at
        }
        for r in result
    ]
    return formatted
