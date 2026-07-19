from typing import List
from sqlalchemy import select
from sqlalchemy.orm import Session
from app.database.models.approved_dataset import ApprovedDataset
from app.database.schemas.approved_dataset import ApprovedDatasetCreate, ApprovedDatasetUpdate
from app.database.repositories.base import BaseRepository


class ApprovedDatasetRepository(BaseRepository[ApprovedDataset, ApprovedDatasetCreate, ApprovedDatasetUpdate]):
    def get_by_dataset(self, db: Session, dataset_id: int) -> List[ApprovedDataset]:
        stmt = select(self.model).where(self.model.dataset_id == dataset_id)
        return list(db.execute(stmt).scalars().all())

    def get_by_approver(self, db: Session, user_id: int) -> List[ApprovedDataset]:
        stmt = select(self.model).where(self.model.approved_by == user_id)
        return list(db.execute(stmt).scalars().all())


approved_dataset = ApprovedDatasetRepository(ApprovedDataset)
