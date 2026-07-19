import sys
import os
from pathlib import Path
from dotenv import load_dotenv

# Force reload dotenv to ensure we have the latest
env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=env_path, override=True)

# Add backend directory to path
backend_dir = str(Path(__file__).resolve().parent.parent / "backend")
if backend_dir not in sys.path:
    sys.path.append(backend_dir)

from app.core.config import settings
from sqlalchemy import create_engine, text

def test_connection():
    print(f"Testing connection to: {settings.DATABASE_URL.split('@')[-1]}")
    try:
        engine = create_engine(settings.DATABASE_URL, pool_pre_ping=True)
        with engine.connect() as conn:
            result = conn.execute(text("SELECT 1")).fetchone()
            if result and result[0] == 1:
                print("Connection successful!")
            else:
                print("Connection failed, SELECT 1 returned unexpected result.")
    except Exception as e:
        print(f"Connection failed: {e}")

if __name__ == "__main__":
    test_connection()
