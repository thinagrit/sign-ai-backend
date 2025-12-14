import tensorflow as tf
import numpy as np
import os

os.makedirs("model_data", exist_ok=True)

# สมมติ input ขนาด 63 (landmark 21 จุด x 3)
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(63,)),
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dense(32, activation="relu"),
    tf.keras.layers.Dense(5, activation="softmax")  # สมมติว่ามี 5 คลาส
])

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

# เทรน mock model ด้วย random data เพื่อให้บันทึกได้
X = np.random.rand(100, 63)
y = np.random.randint(0, 5, size=(100,))
model.fit(X, y, epochs=2)

model.save("model_data/model.h5")

# สร้าง labels.txt
with open("model_data/labels.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(["Hello", "Thanks", "ILoveYou", "Yes", "No"]))

print("✅ Dummy model and labels created.")
