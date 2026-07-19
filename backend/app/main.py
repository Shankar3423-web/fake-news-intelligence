from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.core.config import settings
from app.database.database import engine, Base
import app.database.base # Ensure all models are registered before creating tables
Base.metadata.create_all(bind=engine)

from sqlalchemy import text
try:
    with engine.connect() as conn:
        conn.execute(text("ALTER TABLE feedbacks ADD COLUMN status VARCHAR(50) DEFAULT 'PENDING' NOT NULL"))
        conn.commit()
except Exception:
    pass


app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    debug=settings.DEBUG
)

# Set all CORS enabled origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {
        "message": f"Welcome to {settings.APP_NAME}",
        "version": settings.APP_VERSION,
        "status": "online"
    }

import sys
import os
# Add parent directory to sys.path so we can import 'ml' module
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
# Change the current working directory to the project root to fix relative paths in the ml modules
os.chdir(parent_dir)

from dotenv import load_dotenv
load_dotenv()

# Import and include API routers
from app.api.routes import (
    prediction_router,
    verification_router,
    feedback_router,
    admin_router,
    retraining_router,
    model_router,
    health_router,
    analyze_router,
    auth_router,
)

app.include_router(prediction_router, prefix="/api/v1")
app.include_router(verification_router, prefix="/api/v1")
app.include_router(feedback_router, prefix="/api/v1")
app.include_router(admin_router, prefix="/api/v1")
app.include_router(retraining_router, prefix="/api/v1")
app.include_router(model_router, prefix="/api/v1")
app.include_router(health_router, prefix="/api/v1")
app.include_router(analyze_router, prefix="/api/v1")
app.include_router(auth_router, prefix="/api/v1")
