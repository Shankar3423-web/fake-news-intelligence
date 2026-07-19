from typing import Optional
from pydantic import Field
from app.database.schemas.base import BaseSchema, TimestampedSchema

class UserBase(BaseSchema):
    email: str = Field(..., max_length=255)
    username: str = Field(..., min_length=3, max_length=50)
    is_active: bool = True
    is_superuser: bool = False

class UserCreate(UserBase):
    password: str = Field(..., min_length=6, max_length=128)

class UserUpdate(BaseSchema):
    email: Optional[str] = Field(None, max_length=255)
    username: Optional[str] = Field(None, min_length=3, max_length=50)
    password: Optional[str] = Field(None, min_length=6, max_length=128)
    is_active: Optional[bool] = None
    is_superuser: Optional[bool] = None

class UserResponse(TimestampedSchema, UserBase):
    pass

class UserLogin(BaseSchema):
    email: str = Field(..., max_length=255)
    password: str = Field(..., min_length=6, max_length=128)

class Token(BaseSchema):
    access_token: str
    token_type: str
    user: UserResponse
