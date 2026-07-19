from app.database.database import engine, Base
import app.database.base # Ensure all models are registered before creating tables

def wipe_database():
    print("WARNING: Dropping all tables...")
    Base.metadata.drop_all(bind=engine)
    
    print("Recreating all tables with empty schemas...")
    Base.metadata.create_all(bind=engine)
    
    print("Database wiped perfectly!")

if __name__ == "__main__":
    wipe_database()
