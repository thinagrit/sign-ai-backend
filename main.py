import numpy as np
import cv2
from fastapi import FastAPI, UploadFile, File
import tflite_runtime.interpreter as tflite

app = FastAPI()

# โหลดโมเดล TFLite
interpreter = tflite.Interpreter(model_path="model_data/model.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

@app.get("/")
def root():
    return {"message": "Backend is running (TFLite)"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # อ่านไฟล์ภาพ
    contents = await file.read()
    img_array = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    # เตรียมภาพสำหรับโมเดล
    img = cv2.resize(img, (224, 224))  # ถ้าโมเดลคุณใช้ขนาดอื่น ให้บอกผม
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)

    # ใส่ภาพเข้าโมเดล
    interpreter.set_tensor(input_details[0]['index'], img)
    interpreter.invoke()

    # ดึงผลลัพธ์
    output = interpreter.get_tensor(output_details[0]['index'])
    result = int(np.argmax(output))

    return {"prediction": result}
