from typing import List
from sqlalchemy import select
from sqlalchemy.orm import Session
from app.database.models.verification_queue import VerificationQueue
from app.database.schemas.verification_queue import VerificationQueueCreate, VerificationQueueUpdate
from app.database.repositories.base import BaseRepository


class VerificationQueueRepository(BaseRepository[VerificationQueue, VerificationQueueCreate, VerificationQueueUpdate]):
    def get_by_status(self, db: Session, status: str) -> List[VerificationQueue]:
        stmt = select(self.model).where(self.model.status == status)
        return list(db.execute(stmt).scalars().all())

    def get_assigned_tasks(self, db: Session, user_id: int) -> List[VerificationQueue]:
        stmt = select(self.model).where(self.model.assigned_to == user_id)
        return list(db.execute(stmt).scalars().all())


verification_queue = VerificationQueueRepository(VerificationQueue)
