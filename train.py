import tensorflow as tf
import numpy as np
import os

# ===============================
# 1) โหลด dataset จากโฟลเดอร์
# ===============================

data_dir = "dataset"
img_height = 224
img_width = 224
batch_size = 4

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size,
)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size,
)

class_names = train_ds.class_names
print("Class list =", class_names)

# ===============================
# 2) สร้างโมเดลง่ายๆ
# ===============================

model = tf.keras.Sequential([
    tf.keras.layers.Rescaling(1./255, input_shape=(img_height, img_width, 3)),

    tf.keras.layers.Conv2D(16, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Conv2D(32, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Conv2D(64, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(len(class_names), activation='softmax'),
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# ===============================
# 3) เทรนโมเดล
# ===============================
model.fit(train_ds, validation_data=val_ds, epochs=10)

# ===============================
# 4) บันทึกเป็น .h5
# ===============================
model.save("model.h5")
print("Saved model.h5")

# ===============================
# 5) แปลงเป็น TFLite
# ===============================

converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

with open("model.tflite", "wb") as f:
    f.write(tflite_model)

print("Saved model.tflite")
