"""
Booking Execution Agent
=======================

Takes the collected booking information and:
- Looks up the logged-in user in the DB (first_name, last_name, email)
- Generates a booking_reference and room_number
- Inserts a row into the `bookings` table in Cloud SQL
- Sends confirmation email
"""

import uuid
from datetime import datetime
from typing import Dict, Any

from .state import HotelIQState
from .booking_state import GuestInfo
from .email_service import send_booking_confirmation_email
from logger_config import get_logger

# Use your pooled DB connection
from sql.db_pool import get_connection

logger = get_logger(__name__)


def _generate_booking_reference() -> str:
    """Generate a short human-friendly booking reference like REF-A1B2C3."""
    return "REF-" + uuid.uuid4().hex[:6].upper()


def _generate_room_number(conn, hotel_id: int) -> str:
    """
    Generate a pseudo-sequential room number.

    Pattern:
      - 100–110
      - 200–210
      - 300–310
      - ...
    We cycle through these ranges per hotel based on how many bookings exist.
    """
    floors = list(range(1, 10))  # 1xx, 2xx, ... 9xx
    allowed_rooms = []
    for floor in floors:
        base = floor * 100
        for offset in range(0, 11):  # 0..10  → 100..110
            allowed_rooms.append(base + offset)

    with conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM bookings WHERE hotel_id = %s", (hotel_id,))
        count = cur.fetchone()[0] or 0

    index = count % len(allowed_rooms)
    room_number = allowed_rooms[index]
    return str(room_number)


async def booking_execution_node(state: HotelIQState) -> HotelIQState:
    """
    Finalize the booking:

    - Get user info from `users` table using state["user_id"]
    - Insert into `bookings` table
    - Send confirmation email
    - Respond to the user with booking_reference etc.
    """
    booking_state = state.get("booking_conversation") or {}
    hotel_id_raw = booking_state.get("hotel_id") or state.get("hotel_id")
    hotel_name = booking_state.get("hotel_name") or state.get("metadata", {}).get("hotel_name", "this hotel")
    hotel_info: Dict[str, Any] = booking_state.get("hotel_info") or state.get("metadata", {}).get("hotel_info", {})

    if not hotel_id_raw:
        logger.error("No hotel_id found in state during booking execution")
        response_text = "I’m sorry, I couldn’t identify which hotel to book. Please try again."
    else:
        hotel_id = int(hotel_id_raw)

        # Rebuild GuestInfo from state
        guest_info_raw = booking_state.get("guest_info") or {}
        guest_info = GuestInfo(**guest_info_raw)

        # Pull logged-in user_id from state (set by ChatService)
        user_id = state.get("user_id")
        if not user_id:
            logger.error("No user_id in state during booking execution")
            response_text = "I’m having trouble verifying your account. Please log in again."
        else:
            try:
                with get_connection() as conn:
                    # 1) Look up user in users table
                    with conn.cursor() as cur:
                        cur.execute(
                            """
                            SELECT first_name, last_name, email
                            FROM users
                            WHERE id = %s
                            """,
                            (user_id,),
                        )
                        row = cur.fetchone()

                    if not row:
                        logger.error("User not found in DB for booking", user_id=user_id)
                        response_text = "I couldn’t find your profile in our system. Please register or log in again."
                    else:
                        db_first_name, db_last_name, db_email = row

                        # If GuestInfo still has its own names/emails, we prefer DB values
                        guest_first_name = db_first_name or guest_info.first_name or "Guest"
                        guest_last_name = db_last_name or guest_info.last_name or ""
                        guest_email = db_email or guest_info.email

                        # Ensure GuestInfo used by email has the DB values
                        guest_info.first_name = guest_first_name
                        guest_info.last_name = guest_last_name
                        guest_info.email = guest_email

                        # 2) Generate booking_reference and room_number
                        booking_reference = _generate_booking_reference()
                        room_number = _generate_room_number(conn, hotel_id)

                        # 3) Insert into bookings table (Cloud SQL)
                        check_in_date = guest_info.check_in_date.date()
                        check_out_date = guest_info.check_out_date.date()
                        num_guests = guest_info.num_guests
                        room_type = booking_state.get("room_type") or getattr(guest_info, "room_type", None)

                        with conn.cursor() as cur:
                            cur.execute(
                                """
                                INSERT INTO bookings (
                                    hotel_id,
                                    booking_reference,
                                    room_number,
                                    guest_first_name,
                                    guest_last_name,
                                    guest_email,
                                    check_in_date,
                                    check_out_date,
                                    status,
                                    room_type,
                                    num_guests,
                                    hotel_name
                                )
                                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                                RETURNING id
                                """,
                                (
                                    hotel_id,
                                    booking_reference,
                                    room_number,
                                    guest_first_name,
                                    guest_last_name,
                                    guest_email,
                                    check_in_date,
                                    check_out_date,
                                    "confirmed",
                                    room_type,
                                    num_guests,
                                    hotel_name,
                                ),
                            )
                            booking_id_row = cur.fetchone()
                            db_booking_id = booking_id_row[0] if booking_id_row else None

                        logger.info(
                            "Booking saved to DB",
                            hotel_id=hotel_id,
                            booking_reference=booking_reference,
                            db_booking_id=db_booking_id,
                        )

                        # 4) Send confirmation email (best-effort)
                        try:
                            await send_booking_confirmation_email(
                                guest_info=guest_info,
                                booking_id=booking_reference,
                                hotel_name=hotel_name,
                                hotel_info=hotel_info,
                            )
                        except Exception as e:
                            logger.error(
                                "Failed to send confirmation email",
                                error=str(e),
                                booking_reference=booking_reference,
                            )

                        # 5) Build success response for the chat
                        lines = [
                            "🎉 Your booking is confirmed!\n",
                            f"Hotel: {hotel_name}",
                            f"Booking Reference: `{booking_reference}`",
                            f"Room Number: {room_number}",
                            f"Check-in: {check_in_date.strftime('%B %d, %Y')}",
                            f"Check-out: {check_out_date.strftime('%B %d, %Y')}",
                            f"Guests: {num_guests}",
                        ]

                        if room_type:
                            lines.append(f"Room Type: {room_type}")

                        lines.extend([
                            f"Name: {guest_first_name} {guest_last_name}\n",
                            f"A confirmation email has been sent to {guest_email}.",
                            "Please keep your booking reference handy for check-in and for accessing the concierge portal."
                        ])
                        response_text = "\n".join(lines)
            except Exception as e:
                logger.error("Error during booking execution", error=str(e))
                response_text = "Something went wrong while finalizing your booking. Please try again in a moment."

    # Mark stage as completed
    booking_state["stage"] = "completed"
    state["booking_conversation"] = booking_state

    # Append assistant message
    msgs = state.get("messages", [])
    msgs.append({"role": "assistant", "content": response_text})
    state["messages"] = msgs

    return state