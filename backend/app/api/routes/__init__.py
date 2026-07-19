from .prediction import router as prediction_router
from .verification import router as verification_router
from .feedback import router as feedback_router
from .admin import router as admin_router
from .retraining import router as retraining_router
from .model import router as model_router
from .health import router as health_router
from .analyze import router as analyze_router
from .auth import router as auth_router

__all__ = [
    "prediction_router",
    "verification_router",
    "feedback_router",
    "admin_router",
    "retraining_router",
    "model_router",
    "health_router",
    "analyze_router",
    "auth_router",
]
