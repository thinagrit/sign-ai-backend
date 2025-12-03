from fastapi import FastAPI, UploadFile, File, Form, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy.orm import Session
import os

from database import Base, engine, get_db
import crud
from schemas import PredictionCreate, PredictionOut

# -------------------------
# INIT DB
# -------------------------
Base.metadata.create_all(bind=engine)

app = FastAPI()

# -------------------------
# CORS (สำคัญมาก)
# -------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"message": "Backend Running OK"}


# ============================================================
# TRANSLATE  (Mock model – ให้แปลได้ 1 ท่า)
# ============================================================

class Landmark63(BaseModel):
    points: list[float]   # 63 ค่า


@app.post("/translate")
async def translate(payload: Landmark63, db: Session = Depends(get_db)):

    if len(payload.points) != 63:
        return {"error": "ต้องส่ง landmark 63 ค่า"}

    # --------------------------
    # MOCK MODEL (เวอร์ชันง่ายสุด)
    # --------------------------

    # ถ้าข้อมูลเข้ามาเป็นอะไรก็แปลว่า “hello”
    label = "hello"
    confidence = 1.0

    # บันทึกลง DB
    saved = crud.create_prediction(
        db,
        PredictionCreate(
            label=label,
            confidence=confidence,
            source="translate"
        )
    )

    return {
        "label": label,
        "confidence": confidence,
        "timestamp": saved.created_at
    }


# ============================================================
# SAVE IMAGE (ทำงานอยู่แล้ว)
# ============================================================

@app.post("/save-image")
async def save_image(
    file: UploadFile = File(...),
    label: str = Form(...),
    db: Session = Depends(get_db)
):
    save_dir = os.path.join("data", label)
    os.makedirs(save_dir, exist_ok=True)

    count = len([f for f in os.listdir(save_dir) if f.endswith(".jpg")])
    filename = f"{label}_{count+1:04d}.jpg"
    filepath = os.path.join(save_dir, filename)

    contents = await file.read()
    with open(filepath, "wb") as f:
        f.write(contents)

    saved = crud.create_prediction(
        db,
        PredictionCreate(
            label=label,
            confidence=0.0,
            source="save-image"
        )
    )

    return {
        "message": "saved",
        "path": filepath,
        "id": saved.id
    }


# ============================================================
# DATASET
# ============================================================

@app.get("/dataset", response_model=list[PredictionOut])
def get_dataset(db: Session = Depends(get_db)):
    return crud.get_all_predictions(db)
