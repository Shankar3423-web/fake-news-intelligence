from typing import List
from sqlalchemy import select
from sqlalchemy.orm import Session
from app.database.models.prediction import Prediction
from app.database.schemas.prediction import PredictionCreate, PredictionUpdate
from app.database.repositories.base import BaseRepository


class PredictionRepository(BaseRepository[Prediction, PredictionCreate, PredictionUpdate]):
    def get_by_user(self, db: Session, user_id: int) -> List[Prediction]:
        stmt = select(self.model).where(self.model.user_id == user_id)
        return list(db.execute(stmt).scalars().all())


prediction = PredictionRepository(Prediction)
