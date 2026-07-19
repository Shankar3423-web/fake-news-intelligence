from typing import Optional
from sqlalchemy import select
from sqlalchemy.orm import Session
from app.database.models.dataset import Dataset
from app.database.schemas.dataset import DatasetCreate, DatasetUpdate
from app.database.repositories.base import BaseRepository


class DatasetRepository(BaseRepository[Dataset, DatasetCreate, DatasetUpdate]):
    def get_by_name(self, db: Session, name: str) -> Optional[Dataset]:
        stmt = select(self.model).where(self.model.name == name)
        return db.execute(stmt).scalar_one_or_none()


dataset = DatasetRepository(Dataset)
