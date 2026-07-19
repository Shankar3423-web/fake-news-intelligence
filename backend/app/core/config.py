from pathlib import Path
from pydantic_settings import BaseSettings
from pydantic import ConfigDict

# Locate .env file relative to this file's position (4 levels up from backend/app/core/config.py)
env_path = Path(__file__).resolve().parents[2] / ".env"

class Settings(BaseSettings):
    APP_NAME: str = "Fake News Intelligence System"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = True
    HOST: str = "127.0.0.1"
    PORT: int = 8000
    DATABASE_URL: str = "postgresql://postgres:password@localhost:5432/fake_news_db"
    SECRET_KEY: str = "CHANGE_THIS_LATER"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30

    model_config = ConfigDict(env_file=env_path, case_sensitive=True, extra="ignore")

settings = Settings()
