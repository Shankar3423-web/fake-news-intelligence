from typing import TYPE_CHECKING
from sqlalchemy import String, Integer
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database.database import Base
from app.database.models.mixins import BaseIdMixin

if TYPE_CHECKING:
    from app.database.models.approved_dataset import ApprovedDataset


class Dataset(Base, BaseIdMixin):
    __tablename__ = "datasets"

    name: Mapped[str] = mapped_column(String(255), unique=True, index=True, nullable=False)
    description: Mapped[str | None] = mapped_column(String(1000), nullable=True)
    source_path: Mapped[str] = mapped_column(String(500), nullable=False)
    num_rows: Mapped[int | None] = mapped_column(Integer, nullable=True)

    # Relationships
    approved_records: Mapped[list["ApprovedDataset"]] = relationship(
        "ApprovedDataset", back_populates="dataset", cascade="all, delete-orphan"
    )
