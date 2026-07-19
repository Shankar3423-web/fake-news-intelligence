from typing import TYPE_CHECKING
from sqlalchemy import Text, ForeignKey, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database.database import Base
from app.database.models.mixins import BaseIdMixin

if TYPE_CHECKING:
    from app.database.models.prediction import Prediction
    from app.database.models.user import User


class Feedback(Base, BaseIdMixin):
    __tablename__ = "feedbacks"

    prediction_id: Mapped[int] = mapped_column(
        ForeignKey("predictions.id", ondelete="CASCADE"), nullable=False, index=True
    )
    user_id: Mapped[int | None] = mapped_column(
        ForeignKey("users.id", ondelete="SET NULL"), nullable=True, index=True
    )
    is_correct: Mapped[bool] = mapped_column(nullable=False)
    user_comment: Mapped[str | None] = mapped_column(Text, nullable=True)
    status: Mapped[str] = mapped_column(String(50), default="PENDING", nullable=False, index=True)

    # Relationships
    prediction: Mapped["Prediction"] = relationship("Prediction", back_populates="feedbacks")
    user: Mapped["User | None"] = relationship("User", back_populates="feedbacks")
