from datetime import datetime
from typing import TYPE_CHECKING
from sqlalchemy import String, DateTime, ForeignKey
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database.database import Base
from app.database.models.mixins import BaseIdMixin

if TYPE_CHECKING:
    from app.database.models.prediction import Prediction


class LiveVerification(Base, BaseIdMixin):
    __tablename__ = "live_verifications"

    prediction_id: Mapped[int] = mapped_column(
        ForeignKey("predictions.id", ondelete="CASCADE"), nullable=False, index=True
    )
    fact_checking_source: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    source_url: Mapped[str] = mapped_column(String(1000), nullable=False)
    verdict: Mapped[str] = mapped_column(String(100), nullable=False)
    verification_date: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)

    # Relationships
    prediction: Mapped["Prediction"] = relationship("Prediction", back_populates="live_verifications")
