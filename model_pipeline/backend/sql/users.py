# sql/users.py
from typing import Optional, Dict, Any
from .db_pool import get_connection


def get_user_by_id(user_id: str) -> Optional[Dict[str, Any]]:
    """
    Fetch basic user info from the users table by id.
    """
    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT id, email, first_name, last_name
            FROM users
            WHERE id = %s
            """,
            (user_id,),
        )
        row = cur.fetchone()
        cur.close()

    if not row:
        return None

    return {
        "id": str(row[0]),
        "email": row[1],
        "first_name": row[2],
        "last_name": row[3],
    }
