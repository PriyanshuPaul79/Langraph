import json

from db import pool, init_db

_db_ready = False


def _ensure_db():
    global _db_ready
    if not _db_ready:
        init_db()
        _db_ready = True


def load_history(user_name: str) -> tuple[list[dict], dict]:
    _ensure_db()
    with pool.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT eh.exam, eh.exam_date, eh.target_goal, eh.no_of_hours,
                       s.plan_text, eh.user_profile, eh.created_at, eh.id as exam_history_id
                FROM exam_history eh
                LEFT JOIN schedules s ON s.exam_history_id = eh.id AND s.plan_text IS NOT NULL
                WHERE eh.user_name = %s
                ORDER BY eh.created_at DESC
                """,
                (user_name,),
            )
            columns = [desc[0] for desc in cur.description]
            rows = [dict(zip(columns, r)) for r in cur.fetchall()]
    if not rows:
        return [], {}
    latest_profile = rows[0].get("user_profile")
    if isinstance(latest_profile, str):
        latest_profile = json.loads(latest_profile)
    return rows, latest_profile or {}


def save_goal(user_name: str, goal: dict, user_profile: dict) -> int:
    _ensure_db()
    with pool.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO exam_history (user_name, exam, exam_date, target_goal, no_of_hours, user_profile) VALUES (%s, %s, %s, %s, %s, %s) RETURNING id",
                (
                    user_name,
                    goal["exam"],
                    goal.get("exam_date"),
                    goal.get("target_goal"),
                    goal.get("no_of_hours"),
                    json.dumps(user_profile),
                ),
            )
            conn.commit()
            return cur.fetchone()[0]


def save_plan_text(history_id: int, user_name: str, plan_text: str):
    _ensure_db()
    with pool.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO schedules (user_name, exam_history_id, plan_text) VALUES (%s, %s, %s)",
                (user_name, history_id, plan_text),
            )
            conn.commit()


def save_schedule_task(user_name: str, exam_history_id: int, task_name: str, scheduled_day: str) -> int:
    _ensure_db()
    with pool.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO schedules (user_name, exam_history_id, task_name, scheduled_day, is_completed) VALUES (%s, %s, %s, %s, FALSE) RETURNING id",
                (user_name, exam_history_id, task_name, scheduled_day),
            )
            conn.commit()
            return cur.fetchone()[0]


def load_schedule(user_name: str, exam_history_id: int = None) -> list[dict]:
    _ensure_db()
    with pool.connection() as conn:
        with conn.cursor() as cur:
            if exam_history_id:
                cur.execute(
                    "SELECT id, exam_history_id, task_name, scheduled_day, is_completed, created_at FROM schedules WHERE user_name = %s AND exam_history_id = %s ORDER BY id ASC",
                    (user_name, exam_history_id),
                )
            else:
                cur.execute(
                    "SELECT id, exam_history_id, task_name, scheduled_day, is_completed, created_at FROM schedules WHERE user_name = %s ORDER BY id ASC",
                    (user_name,),
                )
            columns = [desc[0] for desc in cur.description]
            return [dict(zip(columns, r)) for r in cur.fetchall()]


def update_task_completion(task_id: int, is_completed: bool) -> bool:
    _ensure_db()
    with pool.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE schedules SET is_completed = %s WHERE id = %s",
                (is_completed, task_id),
            )
            conn.commit()
            return cur.rowcount > 0
