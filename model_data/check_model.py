with open("model.tflite", "rb") as f:
    header = f.read(4)
    print(header)
