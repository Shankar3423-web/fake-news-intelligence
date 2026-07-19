import sys
from pathlib import Path

# Add backend directory to sys.path
backend_dir = str(Path(__file__).resolve().parent.parent)
if backend_dir not in sys.path:
    sys.path.append(backend_dir)

print("Starting DB setup verification...")

try:
    print("1. Checking config imports...")
    from app.core.config import settings
    print(f"   App name: {settings.APP_NAME}")
    # Mask password for safety
    db_url = settings.DATABASE_URL
    if "@" in db_url:
        parts = db_url.split("@")
        cred_part = parts[0].split("://")
        masked_cred = cred_part[0] + "://****:****"
        masked_url = masked_cred + "@" + parts[1]
    else:
        masked_url = db_url
    print(f"   Database URL: {masked_url}")

    print("2. Checking database core imports...")
    from app.database.database import engine, Base
    from app.database.session import SessionLocal, get_db
    print("   Database engine and session setup imported successfully.")

    print("3. Checking model imports from base...")
    from app.database.base import (
        User, Dataset, ModelVersion, Prediction, 
        LiveVerification, Feedback, VerificationQueue, ApprovedDataset
    )
    print(f"   Successfully imported all 8 models.")
    print(f"   Registered tables: {list(Base.metadata.tables.keys())}")

    print("4. Checking Pydantic schema imports...")
    from app.database.schemas.user import UserCreate, UserResponse
    from app.database.schemas.dataset import DatasetCreate, DatasetResponse
    from app.database.schemas.model_version import ModelVersionCreate, ModelVersionResponse
    from app.database.schemas.prediction import PredictionCreate, PredictionResponse
    from app.database.schemas.live_verification import LiveVerificationCreate, LiveVerificationResponse
    from app.database.schemas.feedback import FeedbackCreate, FeedbackResponse
    from app.database.schemas.verification_queue import VerificationQueueCreate, VerificationQueueResponse
    from app.database.schemas.approved_dataset import ApprovedDatasetCreate, ApprovedDatasetResponse
    print("   Successfully imported all Pydantic schemas.")

    print("5. Checking Repository imports...")
    from app.database.repositories.user import user as user_repo
    from app.database.repositories.dataset import dataset as dataset_repo
    from app.database.repositories.model_version import model_version as model_version_repo
    from app.database.repositories.prediction import prediction as prediction_repo
    from app.database.repositories.live_verification import live_verification as live_verification_repo
    from app.database.repositories.feedback import feedback as feedback_repo
    from app.database.repositories.verification_queue import verification_queue as verification_queue_repo
    from app.database.repositories.approved_dataset import approved_dataset as approved_dataset_repo
    print("   Successfully imported all repository instances.")

    print("\nSUCCESS: All database layer files imported and validated without errors!")
except Exception as e:
    print(f"\nERROR: Verification failed with: {str(e)}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
