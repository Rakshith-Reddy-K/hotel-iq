# sql/bookings.py
from typing import Dict, Any
from datetime import datetime
from .db_pool import get_connection


def insert_booking(booking: Dict[str, Any]) -> None:
    """
    Insert a booking record into the bookings table.
    booking dict is expected to contain:

    {
        "booking_id": str,
        "user_id": str,
        "first_name": str,
        "last_name": str,
        "email": str,
        "hotel_id": str,
        "hotel_name": str,
        "room_type": str,
        "check_in_date": datetime,
        "check_out_date": datetime,
        "num_guests": int,
    }
    """
    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO bookings (
                id,
                user_id,
                first_name,
                last_name,
                email,
                hotel_id,
                hotel_name,
                room_type,
                check_in_date,
                check_out_date,
                num_guests,
                created_at
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """,
            (
                booking["booking_id"],
                booking["user_id"],
                booking["first_name"],
                booking["last_name"],
                booking["email"],
                booking["hotel_id"],
                booking["hotel_name"],
                booking["room_type"],
                booking["check_in_date"],
                booking["check_out_date"],
                booking["num_guests"],
                datetime.utcnow(),
            ),
        )
        cur.close()
