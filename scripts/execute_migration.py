import sys
import os
from pathlib import Path
from dotenv import load_dotenv

env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=env_path, override=True)

backend_dir = str(Path(__file__).resolve().parent.parent / "backend")
if backend_dir not in sys.path:
    sys.path.append(backend_dir)

import alembic.config
from sqlalchemy import create_engine, text, MetaData
from sqlalchemy.orm import sessionmaker

def test_connection(engine, name):
    print(f"Testing {name} connection...")
    try:
        with engine.connect() as conn:
            result = conn.execute(text("SELECT 1")).fetchone()
            if result and result[0] == 1:
                print(f"✅ {name} connection successful!")
                return True
    except Exception as e:
        print(f"❌ {name} connection failed: {e}")
        return False
    return False

def run_alembic_upgrade():
    print("\nRunning Alembic migrations on Render database...")
    alembic_ini_path = Path(__file__).resolve().parent.parent / "alembic.ini"
    alembic_args = [
        '-c', str(alembic_ini_path),
        'upgrade', 'head'
    ]
    try:
        alembic.config.main(argv=alembic_args)
        print("✅ Alembic migrations applied successfully.")
    except Exception as e:
        print(f"❌ Alembic migration failed: {e}")
        sys.exit(1)

def get_topological_sorted_tables(metadata):
    """Sort tables to respect foreign keys (parents before children)."""
    return [table for table in metadata.sorted_tables]

def migrate_data(local_engine, render_engine):
    print("\nStarting data migration...")
    
    local_metadata = MetaData()
    local_metadata.reflect(bind=local_engine)
    
    render_metadata = MetaData()
    render_metadata.reflect(bind=render_engine)
    
    tables = get_topological_sorted_tables(local_metadata)
    
    with render_engine.connect() as render_conn:
        with local_engine.connect() as local_conn:
            with render_conn.begin():
                # Disable FK checks temporarily if needed, but topological sort should handle inserts.
                # However, for deletions (if we want to clear render first), we should reverse.
                
                # We assume Render DB is fresh and empty due to alembic upgrade.
                
                for table in tables:
                    table_name = table.name
                    if table_name == 'alembic_version':
                        continue # Skip alembic version, already handled by upgrade
                        
                    print(f"Migrating table: {table_name}")
                    
                    # Read all rows from local
                    result = local_conn.execute(table.select()).fetchall()
                    if not result:
                        print(f"  - No records to migrate in {table_name}")
                        continue
                        
                    # Insert into Render
                    render_table = render_metadata.tables[table_name]
                    
                    # Convert to list of dicts for bulk insert
                    # Fetch keys from the result
                    keys = result[0]._mapping.keys()
                    records = [dict(zip(keys, row)) for row in result]
                    
                    render_conn.execute(render_table.insert(), records)
                    print(f"  - Migrated {len(records)} records for {table_name}")
                    
                    # Update sequence if postgres (reset primary key sequence)
                    # We check if there's an autoincrement primary key (usually named id)
                    try:
                        pk = [c for c in table.primary_key.columns][0]
                        if pk.autoincrement and pk.type.python_type == int:
                            seq_sql = f"SELECT setval(pg_get_serial_sequence('{table_name}', '{pk.name}'), coalesce(max({pk.name}), 1), max({pk.name}) IS NOT NULL) FROM {table_name};"
                            render_conn.execute(text(seq_sql))
                            print(f"  - Reset sequence for {table_name}.{pk.name}")
                    except Exception as e:
                        print(f"  - (Note: Could not reset sequence for {table_name}: {e})")
                        
    print("✅ Data migration complete!")

def verify_migration(local_engine, render_engine):
    print("\nVerifying migration...")
    local_metadata = MetaData()
    local_metadata.reflect(bind=local_engine)
    
    render_metadata = MetaData()
    render_metadata.reflect(bind=render_engine)
    
    success = True
    
    with local_engine.connect() as local_conn, render_engine.connect() as render_conn:
        for table in local_metadata.sorted_tables:
            table_name = table.name
            
            local_count = local_conn.execute(text(f'SELECT COUNT(*) FROM "{table_name}"')).scalar()
            render_count = render_conn.execute(text(f'SELECT COUNT(*) FROM "{table_name}"')).scalar()
            
            if local_count == render_count:
                print(f"✅ {table_name}: {local_count} rows match.")
            else:
                print(f"❌ {table_name}: Local has {local_count}, Render has {render_count}.")
                success = False
                
    if success:
        print("\n🎉 Migration verified successfully!")
    else:
        print("\n⚠️ Migration verification failed on some tables.")

def main():
    local_url = os.environ.get("LOCAL_DATABASE_URL")
    render_url = os.environ.get("DATABASE_URL")
    
    if not local_url or not render_url:
        print("Missing LOCAL_DATABASE_URL or DATABASE_URL in .env")
        sys.exit(1)
        
    local_engine = create_engine(local_url, pool_pre_ping=True)
    render_engine = create_engine(render_url, pool_pre_ping=True)
    
    if not test_connection(local_engine, "Local"):
        sys.exit(1)
    if not test_connection(render_engine, "Render"):
        sys.exit(1)
        
    run_alembic_upgrade()
    migrate_data(local_engine, render_engine)
    verify_migration(local_engine, render_engine)

if __name__ == "__main__":
    main()
