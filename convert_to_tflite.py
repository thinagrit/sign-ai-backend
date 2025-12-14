import tensorflow as tf

# โหลดโมเดล .h5
model = tf.keras.models.load_model("model_data/model.h5")

# แปลงเป็น TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# เซฟไฟล์ .tflite
with open("model_data/model.tflite", "wb") as f:
    f.write(tflite_model)

print("แปลงเป็น model.tflite สำเร็จ!")
