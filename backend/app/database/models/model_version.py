from typing import TYPE_CHECKING
from sqlalchemy import String, Float, JSON, Boolean
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database.database import Base
from app.database.models.mixins import BaseIdMixin

if TYPE_CHECKING:
    from app.database.models.prediction import Prediction


class ModelVersion(Base, BaseIdMixin):
    __tablename__ = "model_versions"

    version_str: Mapped[str] = mapped_column(String(50), unique=True, index=True, nullable=False)
    model_name: Mapped[str] = mapped_column(String(255), nullable=False)
    algorithm_name: Mapped[str] = mapped_column(String(255), nullable=False)
    accuracy: Mapped[float | None] = mapped_column(Float, nullable=True)
    f1_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    parameters: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    filepath: Mapped[str] = mapped_column(String(500), nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)

    # Relationships
    predictions: Mapped[list["Prediction"]] = relationship(
        "Prediction", back_populates="model_version"
    )
