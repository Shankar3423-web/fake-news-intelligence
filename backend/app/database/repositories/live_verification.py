from typing import List
from sqlalchemy import select
from sqlalchemy.orm import Session
from app.database.models.live_verification import LiveVerification
from app.database.schemas.live_verification import LiveVerificationCreate, LiveVerificationUpdate
from app.database.repositories.base import BaseRepository


class LiveVerificationRepository(BaseRepository[LiveVerification, LiveVerificationCreate, LiveVerificationUpdate]):
    def get_by_prediction(self, db: Session, prediction_id: int) -> List[LiveVerification]:
        stmt = select(self.model).where(self.model.prediction_id == prediction_id)
        return list(db.execute(stmt).scalars().all())


live_verification = LiveVerificationRepository(LiveVerification)
