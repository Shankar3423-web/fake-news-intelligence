from typing import Optional
from sqlalchemy import select
from sqlalchemy.orm import Session
from app.database.models.model_version import ModelVersion
from app.database.schemas.model_version import ModelVersionCreate, ModelVersionUpdate
from app.database.repositories.base import BaseRepository


class ModelVersionRepository(BaseRepository[ModelVersion, ModelVersionCreate, ModelVersionUpdate]):
    def get_by_version(self, db: Session, version_str: str) -> Optional[ModelVersion]:
        stmt = select(self.model).where(self.model.version_str == version_str)
        return db.execute(stmt).scalar_one_or_none()

    def get_active(self, db: Session) -> Optional[ModelVersion]:
        stmt = select(self.model).where(self.model.is_active == True)
        return db.execute(stmt).scalars().first()


model_version = ModelVersionRepository(ModelVersion)
