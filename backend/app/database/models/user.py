from typing import TYPE_CHECKING
from sqlalchemy import String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database.database import Base
from app.database.models.mixins import BaseIdMixin

if TYPE_CHECKING:
    from app.database.models.prediction import Prediction
    from app.database.models.feedback import Feedback
    from app.database.models.approved_dataset import ApprovedDataset
    from app.database.models.verification_queue import VerificationQueue


class User(Base, BaseIdMixin):
    __tablename__ = "users"

    email: Mapped[str] = mapped_column(String(255), unique=True, index=True, nullable=False)
    username: Mapped[str] = mapped_column(String(255), unique=True, index=True, nullable=False)
    hashed_password: Mapped[str] = mapped_column(String(255), nullable=False)
    is_active: Mapped[bool] = mapped_column(default=True, nullable=False)
    is_superuser: Mapped[bool] = mapped_column(default=False, nullable=False)

    # Relationships
    predictions: Mapped[list["Prediction"]] = relationship(
        "Prediction", back_populates="user", cascade="all, delete-orphan"
    )
    feedbacks: Mapped[list["Feedback"]] = relationship(
        "Feedback", back_populates="user"
    )
    approved_datasets: Mapped[list["ApprovedDataset"]] = relationship(
        "ApprovedDataset", back_populates="approver"
    )
    assigned_tasks: Mapped[list["VerificationQueue"]] = relationship(
        "VerificationQueue", back_populates="assigned_user"
    )
