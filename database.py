from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
import os


# --------------------------------------------------------
# DATABASE URL (รองรับ SQLite + PostgreSQL)
# --------------------------------------------------------

# 1) ถ้ามีตัวแปร ENV → ใช้ PostgreSQL (เหมาะสำหรับ Render)
DATABASE_URL = os.getenv("DATABASE_URL")

# 2) ถ้าไม่มี ENV → ใช้ SQLite (สำหรับ run ในเครื่อง)
if not DATABASE_URL:
    DATABASE_URL = "sqlite:///./signai.db"


# --------------------------------------------------------
# CREATE ENGINE
# --------------------------------------------------------
if DATABASE_URL.startswith("sqlite"):
    engine = create_engine(
        DATABASE_URL,
        connect_args={"check_same_thread": False},
        pool_pre_ping=True
    )
else:
    # PostgreSQL
    engine = create_engine(
        DATABASE_URL,
        pool_pre_ping=True,        # แก้ connection timeout บน Render
        pool_size=10,
        max_overflow=20
    )


# --------------------------------------------------------
# SESSION
# --------------------------------------------------------
SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine
)


# --------------------------------------------------------
# BASE MODEL
# --------------------------------------------------------
Base = declarative_base()


# --------------------------------------------------------
# DEPENDENCY (ใช้ใน main.py)
# --------------------------------------------------------
def get_db():
    db = SessionLocal()
    try:
        yield db
    except Exception as e:
        print("Database session error:", e)
        raise e
    finally:
        db.close()
