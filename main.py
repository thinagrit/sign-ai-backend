import numpy as np
import json
from fastapi import FastAPI, UploadFile, File, Depends
import tensorflow as tf
from fastapi.middleware.cors import CORSMiddleware
from database import SessionLocal, engine
import models
import crud

# สร้างตาราง DB ถ้ายังไม่มี
models.Base.metadata.create_all(bind=engine)

app = FastAPI()

# อนุญาตให้ frontend เรียกได้
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# โหลด labels
with open("model_data/labels.txt", "r", encoding="utf-8") as f:
    LABELS = f.read().splitlines()

# โหลดโมเดล TFLite ด้วย TensorFlow (ไม่ใช้ tflite-runtime)
interpreter = tf.lite.Interpreter(model_path="model_data/model.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()


# Dependency DB
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@app.get("/")
def root():
    return {"message": "Backend is running (TFLite via TensorFlow)", "labels": LABELS}


@app.post("/predict")
async def predict(file: UploadFile = File(...), db=Depends(get_db)):

    raw = await file.read()

    try:
        data = json.loads(raw.decode("utf-8"))
        arr = np.array(data).astype("float32")
    except:
        return {"error": "ต้องส่ง JSON array ยาว 63 ค่า"}

    if arr.shape[0] != 63:
        return {"error": "input ต้องยาว 63 ค่า"}

    arr = np.expand_dims(arr, axis=0)

    # ใส่เข้าโมเดล
    interpreter.set_tensor(input_details[0]['index'], arr)
    interpreter.invoke()

    output = interpreter.get_tensor(output_details[0]['index'])

    class_id = int(np.argmax(output))
    confidence = float(np.max(output))
    label = LABELS[class_id]

    # บันทึก DB
    crud.save_prediction(db, label, confidence, "landmark-63")

    return {
        "label": label,
        "class_id": class_id,
        "confidence": round(confidence, 4)
    }


@app.get("/history")
def history(db=Depends(get_db)):
    records = crud.get_history(db)
    return [
        {
            "id": r.id,
            "label": r.label,
            "confidence": r.confidence,
            "source": r.source,
            "created_at": r.created_at
        }
        for r in records
    ]
