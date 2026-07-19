from typing import Optional
from sqlalchemy import select
from sqlalchemy.orm import Session
from app.database.models.user import User
from app.database.schemas.user import UserCreate, UserUpdate
from app.database.repositories.base import BaseRepository


class UserRepository(BaseRepository[User, UserCreate, UserUpdate]):
    def get_by_email(self, db: Session, email: str) -> Optional[User]:
        stmt = select(self.model).where(self.model.email == email)
        return db.execute(stmt).scalar_one_or_none()

    def get_by_username(self, db: Session, username: str) -> Optional[User]:
        stmt = select(self.model).where(self.model.username == username)
        return db.execute(stmt).scalar_one_or_none()


user = UserRepository(User)
