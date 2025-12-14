import tensorflow as tf
import numpy as np
import os

# สร้างโฟลเดอร์ model_data ถ้ายังไม่มี
os.makedirs("model_data", exist_ok=True)

# Input = 63 ค่า (landmark 21 จุด × 3)
# Output = 5 classes
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(63,)),
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dense(32, activation="relu"),
    tf.keras.layers.Dense(5, activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# เทรน dummy data (เพื่อให้โมเดลเซฟได้)
X = np.random.rand(200, 63)
y = np.random.randint(0, 5, size=(200,))

model.fit(X, y, epochs=5)

# เซฟโมเดล .h5
model.save("model_data/model.h5")

print("สร้าง model.h5 สำเร็จ!")
