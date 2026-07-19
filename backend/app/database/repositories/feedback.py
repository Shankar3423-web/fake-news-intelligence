from typing import List
from sqlalchemy import select
from sqlalchemy.orm import Session
from app.database.models.feedback import Feedback
from app.database.schemas.feedback import FeedbackCreate, FeedbackUpdate
from app.database.repositories.base import BaseRepository


class FeedbackRepository(BaseRepository[Feedback, FeedbackCreate, FeedbackUpdate]):
    def get_by_prediction(self, db: Session, prediction_id: int) -> List[Feedback]:
        stmt = select(self.model).where(self.model.prediction_id == prediction_id)
        return list(db.execute(stmt).scalars().all())


feedback = FeedbackRepository(Feedback)
