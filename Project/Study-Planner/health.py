from dotenv import load_dotenv
load_dotenv()

from psycopg_pool import ConnectionPool
import os
from db import init_db

pool = ConnectionPool(
    os.getenv("DB_URL"),
    min_size=1,
    max_size=5,
    open=False,
    kwargs={"autocommit": True},
)

def check_connection():
    pool.open()
    pool.wait(timeout=10)  # blocks until min_size connections are actually established, raises PoolTimeout if it can't
    
    print("Initializing database and tables...")
    init_db()
    
    with pool.connection() as conn:
        result = conn.execute("SELECT version(), current_database(), now()").fetchone()
        print("✅ Connected successfully")
        print("Postgres version:", result[0])
        print("Database:", result[1])
        print("Server time:", result[2])
        
        # Verify tables exist
        with conn.cursor() as cur:
            cur.execute("""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = 'public'
            """)
            tables = [r[0] for r in cur.fetchall()]
            print("Tables in public schema:", tables)

if __name__ == "__main__":
    check_connection()