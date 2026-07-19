from dotenv import load_dotenv

load_dotenv()

from psycopg_pool import ConnectionPool
import os

pool = ConnectionPool(os.getenv("DB_URL"), min_size=1, max_size=5)


def init_db():
    with pool.connection() as conn:
        conn.autocommit = True
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS exam_history (
                id SERIAL PRIMARY KEY,
                user_name TEXT NOT NULL,
                exam TEXT NOT NULL,
                exam_date TEXT,
                target_goal TEXT,
                no_of_hours INTEGER,
                user_profile JSONB DEFAULT '{}',
                created_at TIMESTAMP DEFAULT NOW()
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS schedules (
                id SERIAL PRIMARY KEY,
                user_name TEXT NOT NULL,
                exam_history_id INTEGER REFERENCES exam_history(id) ON DELETE CASCADE,
                task_name TEXT,
                scheduled_day TEXT,
                is_completed BOOLEAN DEFAULT FALSE,
                created_at TIMESTAMP DEFAULT NOW()
            )
            """
        )
        # ponytail: one-off migration for plan_text column added later
        conn.execute(
            "ALTER TABLE schedules ADD COLUMN IF NOT EXISTS plan_text TEXT"
        )
        conn.execute(
            "ALTER TABLE schedules ALTER COLUMN task_name DROP NOT NULL"
        )
        conn.execute(
            "ALTER TABLE schedules ALTER COLUMN scheduled_day DROP NOT NULL"
        )
        # ponytail: clean up study_plan from exam_history if it still exists from old schema
        conn.execute(
            "ALTER TABLE exam_history DROP COLUMN IF EXISTS study_plan"
        )
