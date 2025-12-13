"""
Booking Collection Agent
========================

Handles natural conversation to collect guest information for booking.
"""

from typing import Dict, Any, List
from datetime import datetime
import re
from dateutil import parser as date_parser

from .state import HotelIQState
from .booking_state import GuestInfo, BookingConversationState
from .config import llm
from .prompt_loader import get_prompts
from logger_config import get_logger
from sql.db_pool import get_connection

logger = get_logger(__name__)


# def parse_date_flexible(date_str: str) -> datetime:
#     """
#     Parse date from various formats flexibly.
    
#     Handles formats like:
#     - 12/25, 12-25, 12/25/2024
#     - December 25, Dec 25
#     - 25th December
#     - tomorrow, next week, etc.
#     """
#     try:
#         parsed_date = date_parser.parse(date_str, fuzzy=True)

#         # If year is not specified, assume current year or next year
#         now = datetime.now()
#         if parsed_date.year == now.year and parsed_date < now:
#             parsed_date = parsed_date.replace(year=parsed_date.year + 1)

#         return parsed_date
#     except Exception as e:
#         logger.warning("Failed to parse date", date_str=date_str, error=str(e))
#         raise ValueError(f"Could not understand date: {date_str}")
def parse_date_flexible(date_str: str) -> datetime:
    """
    Parse date from various formats flexibly.
    
    Handles formats like:
    - 12/25, 12-25, 12/25/2024
    - December 25, Dec 25
    - 25th December
    - tomorrow, next week, etc.
    """
    try:
        parsed_date = date_parser.parse(date_str, fuzzy=True)

        # If year is not specified and date appears to be from a past month,
        # assume next year
        now = datetime.now()
        
        # Only adjust year if the year wasn't explicitly provided in the string
        # and the parsed date is more than 30 days in the past
        if parsed_date.year == now.year:
            days_diff = (parsed_date - now).days
            # If the date is significantly in the past (more than 30 days), assume next year
            if days_diff < -30:
                parsed_date = parsed_date.replace(year=parsed_date.year + 1)

        return parsed_date
    except Exception as e:
        logger.warning("Failed to parse date", date_str=date_str, error=str(e))
        raise ValueError(f"Could not understand date: {date_str}")
def extract_guest_info_from_message(message: str, current_info: GuestInfo) -> tuple[GuestInfo, list[str]]:
    """
    Extract guest information from user message using LLM.

    NOTE (updated behaviour):
    - We will only *use* check-in date, check-out date, and num_guests for flow logic.
    - Name/email/phone may still be extracted but we do NOT ask for them
      and we don't rely on them to consider the booking "complete".
    
    Returns:
        Tuple of (updated GuestInfo, list of extracted field names)
    """
    prompts = get_prompts()

    extraction_prompt = prompts.format(
        "booking_collection.extract_info",
        message=message,
        current_info=current_info.model_dump_json()
    )

    try:
        response = llm.invoke(extraction_prompt)
        import json
        extracted_data = json.loads(response.content)

        updated_fields: list[str] = []

        # Keep these for backward compatibility, but we don't rely on them.
        if extracted_data.get("first_name") and not current_info.first_name:
            current_info.first_name = extracted_data["first_name"]
            updated_fields.append("first_name")

        if extracted_data.get("last_name") and not current_info.last_name:
            current_info.last_name = extracted_data["last_name"]
            updated_fields.append("last_name")

        if extracted_data.get("email") and not current_info.email:
            try:
                current_info.email = extracted_data["email"]
                updated_fields.append("email")
            except ValueError:
                pass

        if extracted_data.get("phone") and not current_info.phone:
            try:
                current_info.phone = extracted_data["phone"]
                updated_fields.append("phone")
            except ValueError:
                pass

        if extracted_data.get("check_in_date") and not current_info.check_in_date:
            try:
                current_info.check_in_date = parse_date_flexible(
                    extracted_data["check_in_date"]
                )
                updated_fields.append("check_in_date")
            except ValueError:
                pass

        if extracted_data.get("check_out_date") and not current_info.check_out_date:
            try:
                current_info.check_out_date = parse_date_flexible(
                    extracted_data["check_out_date"]
                )
                updated_fields.append("check_out_date")
            except ValueError:
                pass

        if extracted_data.get("num_guests") and not current_info.num_guests:
            try:
                current_info.num_guests = int(extracted_data["num_guests"])
                updated_fields.append("num_guests")
            except ValueError:
                pass

        # Optional: if your extraction returns room_type and GuestInfo has it
        if extracted_data.get("room_type") and hasattr(current_info, "room_type") and not getattr(current_info, "room_type", None):
            current_info.room_type = extracted_data["room_type"]
            updated_fields.append("room_type")

        return current_info, updated_fields

    except Exception as e:
        logger.error("Failed to extract info with LLM", error=str(e))
        return current_info, []


# -------------------- NEW HELPERS FOR ROOM TYPES -------------------- #

def _get_hotel_info_from_state(state: HotelIQState) -> Dict[str, Any]:
    """
    Get hotel info from state/booking_conversation.
    This reuses your existing metadata/booking structure.
    """
    # Prefer booking_conversation if present
    if "booking_conversation" in state:
        bc = state["booking_conversation"]
        if isinstance(bc, dict) and bc.get("hotel_info"):
            return bc["hotel_info"]

    # Fallback to metadata set by metadata_agent
    if "metadata" in state:
        meta = state["metadata"] or {}
        if isinstance(meta, dict) and meta.get("hotel_info"):
            return meta["hotel_info"]

    return {}


def _extract_room_types(hotel_id: str) -> List[str]:
    """
    Fetch room type names from the database for a given hotel.
    
    Args:
        hotel_id: The ID of the hotel to fetch room types for.
        
    Returns:
        List of room type names, or a generic fallback list if none found.
    """
    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT room_type
                    FROM rooms
                    WHERE hotel_id = %s
                    """,
                    (hotel_id,),
                )
                rows = cur.fetchall()
                
                if rows:
                    return [row[0] for row in rows]
                    
    except Exception as e:
        pass

    return [
        "Standard King Room",
        "Standard Queen Room",
        "Deluxe King Room",
        "Deluxe Queen Room",
        "Suite",
    ]

def _fuzzy_pick_room_type(user_text: str, options: List[str]) -> str | None:
    """
    Try to match the user's message to one of the room types.
    """
    text = user_text.lower()

    # Exact substring
    for opt in options:
        if opt.lower() in text:
            return opt

    # Partial word/keyword match
    for opt in options:
        words = [w for w in opt.lower().split() if len(w) > 3]
        if any(w in text for w in words):
            return opt

    return None


def _booking_details_complete(guest_info: GuestInfo) -> bool:
    """
    New notion of 'complete' for this flow:
    We ONLY require:
    - check_in_date
    - check_out_date
    - num_guests

    Name/email are now assumed to come from login, not chat.
    """
    return bool(
        guest_info.check_in_date
        and guest_info.check_out_date
        and guest_info.num_guests
    )


def _missing_core_fields(guest_info: GuestInfo) -> List[str]:
    """
    Compute missing fields for the *conversation*.
    We no longer ask for name/email/phone here.
    """
    missing: List[str] = []
    if not guest_info.check_in_date:
        missing.append("check-in date")
    if not guest_info.check_out_date:
        missing.append("check-out date")
    if not guest_info.num_guests:
        missing.append("number of guests")
    return missing


# def _validate_booking_dates(guest_info: GuestInfo) -> tuple[bool, str | None]:
#     """
#     Validate booking dates against business rules.
    
#     Rules:
#     1. Check-in date must be in the future (not in the past)
#     2. Cannot book more than 3 months (90 days) in advance
#     3. Stay duration cannot exceed 14 days
    
#     Returns:
#         Tuple of (is_valid, error_message)
#         If valid: (True, None)
#         If invalid: (False, error_message)
#     """
#     from datetime import timedelta
    
#     if not guest_info.check_in_date or not guest_info.check_out_date:
#         return True, None  # Not enough info to validate yet
    
#     now = datetime.now()
#     today = now.replace(hour=0, minute=0, second=0, microsecond=0)
    
#     check_in = guest_info.check_in_date
#     check_out = guest_info.check_out_date
    
#     # Remove timezone info if present for comparison
#     if check_in.tzinfo:
#         check_in = check_in.replace(tzinfo=None)
#     if check_out.tzinfo:
#         check_out = check_out.replace(tzinfo=None)
    
#     # Rule 1: Check-in date cannot be in the past
#     if check_in.date() < today.date():
#         return False, (
#             f"❌ <b>Check-in date cannot be in the past.</b>\n\n"
#             f"You selected {check_in.strftime('%B %d, %Y')}, but today is {today.strftime('%B %d, %Y')}.\n"
#             f"Please provide a check-in date that is today or in the future."
#         )
    
#     # Rule 2: Cannot book more than 3 months (90 days) in advance
#     max_advance_date = today + timedelta(days=90)
#     if check_in.date() > max_advance_date.date():
#         return False, (
#             f"❌ <b>Cannot book more than 3 months in advance.</b>\n\n"
#             f"You're trying to check in on {check_in.strftime('%B %d, %Y')}, "
#             f"but we can only accept bookings up to {max_advance_date.strftime('%B %d, %Y')}.\n"
#             f"Please choose a check-in date within the next 3 months."
#         )
    
#     # Rule 3: Stay duration cannot exceed 14 days
#     stay_duration = (check_out - check_in).days
#     if stay_duration > 14:
#         return False, (
#             f"❌ <b>Maximum stay is 14 days.</b>\n\n"
#             f"Your selected dates ({check_in.strftime('%B %d, %Y')} to {check_out.strftime('%B %d, %Y')}) "
#             f"would be a {stay_duration}-day stay.\n"
#             f"Please adjust your dates to stay within the 14-day maximum."
#         )
    
#     # Additional validation: Check-out must be after check-in
#     if check_out <= check_in:
#         return False, (
#             f"❌ <b>Check-out date must be after check-in date.</b>\n\n"
#             f"Check-in: {check_in.strftime('%B %d, %Y')}\n"
#             f"Check-out: {check_out.strftime('%B %d, %Y')}\n\n"
#             f"Please provide valid dates where check-out is after check-in."
#         )
    
#     return True, None

def _validate_booking_dates(guest_info: GuestInfo) -> tuple[bool, str | None]:
    """
    Validate booking dates against business rules.
    
    Rules:
    1. Check-in date must be in the future (not in the past)
    2. Cannot book more than 3 months (90 days) in advance
    3. Stay duration cannot exceed 14 days
    
    Returns:
        Tuple of (is_valid, error_message)
        If valid: (True, None)
        If invalid: (False, error_message)
    """
    from datetime import timedelta
    
    if not guest_info.check_in_date or not guest_info.check_out_date:
        return True, None  # Not enough info to validate yet
    
    now = datetime.now()
    today = now.replace(hour=0, minute=0, second=0, microsecond=0)
    
    check_in = guest_info.check_in_date
    check_out = guest_info.check_out_date
    
    # Remove timezone info if present for comparison
    if check_in.tzinfo:
        check_in = check_in.replace(tzinfo=None)
    if check_out.tzinfo:
        check_out = check_out.replace(tzinfo=None)
    
    # Additional validation: Check-out must be after check-in (check this first)
    if check_out <= check_in:
        return False, (
            f"❌ **Check-out date must be after check-in date.**\n\n"
            f"Check-in: {check_in.strftime('%B %d, %Y')}\n"
            f"Check-out: {check_out.strftime('%B %d, %Y')}\n\n"
            f"Please provide valid dates where check-out is after check-in."
        )
    
    # Rule 1: Check-in date cannot be in the past
    if check_in.date() < today.date():
        return False, (
            f"❌ **Check-in date cannot be in the past.**\n\n"
            f"You selected {check_in.strftime('%B %d, %Y')}, but today is {today.strftime('%B %d, %Y')}.\n"
            f"Please provide a check-in date that is today or in the future."
        )
    
    # Rule 2: Cannot book more than 3 months (90 days) in advance
    max_advance_date = today + timedelta(days=90)
    if check_in.date() > max_advance_date.date():
        return False, (
            f"❌ **Cannot book more than 3 months in advance.**\n\n"
            f"You're trying to check in on {check_in.strftime('%B %d, %Y')}, "
            f"but we can only accept bookings up to {max_advance_date.strftime('%B %d, %Y')}.\n"
            f"Please choose a check-in date within the next 3 months."
        )
    
    # Rule 3: Stay duration cannot exceed 14 days
    stay_duration = (check_out - check_in).days
    if stay_duration > 14:
        return False, (
            f"❌ **Maximum stay is 14 days.**\n\n"
            f"Your selected dates ({check_in.strftime('%B %d, %Y')} to {check_out.strftime('%B %d, %Y')}) "
            f"would be a {stay_duration}-day stay.\n"
            f"Please adjust your dates to stay within the 14-day maximum."
        )
    
    return True, None
# -------------------- MAIN BOOKING COLLECTION NODE -------------------- #

async def booking_collection_node(state: HotelIQState) -> HotelIQState:
    """
    Booking Collection Agent: Manages conversation to collect guest information.

    UPDATED STAGES:
    - initial: user just requested booking; greet & show room types
    - choosing_room_type: user picks a room type
    - collecting: gather check-in, check-out, num guests
    - confirming: final confirmation before execution
    - executing: handled by booking_execution_agent
    """
    user_message = state["messages"][-1]["content"]
    prompts = get_prompts()
    hotel_id = state["hotel_id"]
    # Initialize booking state if not exists
    if "booking_conversation" not in state:
        hotel_info = _get_hotel_info_from_state(state)
        state["booking_conversation"] = {
            "stage": "initial",
            "guest_info": GuestInfo().model_dump(),
            "hotel_id": state.get("hotel_id", ""),
            "hotel_name": state.get("metadata", {}).get("hotel_name", "this hotel"),
            "hotel_info": hotel_info,
            "confirmation_pending": False,
            "selected_room_type": None,
            "room_type": None,          # <--- important for execution agent
            "available_room_types": [],
        }

    booking_state = state["booking_conversation"]
    guest_info = GuestInfo(**booking_state["guest_info"])
    stage = booking_state["stage"]
    hotel_name = booking_state["hotel_name"]
    hotel_info = booking_state.get("hotel_info") or _get_hotel_info_from_state(state)
    booking_state["hotel_info"] = hotel_info  # keep it in sync

    logger.info(
        "Booking Collection Agent",
        stage=stage,
        hotel_name=hotel_name,
    )

    # -------------------- STAGE: INITIAL -------------------- #
    if stage == "initial":
        # Get room types for this hotel
        room_types = _extract_room_types(hotel_id)
        booking_state["available_room_types"] = room_types

        # Use your existing initial prompt as intro (so tone stays same)
        intro = prompts.format(
            "booking_collection.initial_confirmation",
            hotel_name=hotel_name,
        )

        room_list_str = "\n".join(f"- {rt}" for rt in room_types)

        response = (
            f"{intro}\n\n"
            f"Before we proceed, here are the room types available at <b>{hotel_name}</b>:\n"
            f"{room_list_str}\n\n"
            f"👉 Which room type would you like to book?"
        )

        booking_state["stage"] = "choosing_room_type"
        booking_state["confirmation_pending"] = False  # no yes/no here anymore

    # -------------------- STAGE: CHOOSING ROOM TYPE -------------------- #
    elif stage == "choosing_room_type":
        room_types = booking_state.get("available_room_types") or _extract_room_types(
            hotel_info
        )

        chosen = _fuzzy_pick_room_type(user_message, room_types)
        if not chosen:
            room_list_str = "\n".join(f"- {rt}" for rt in room_types)
            response = (
                f"Got it! Just to confirm, which exact room type would you like at <b>{hotel_name}</b>?\n\n"
                f"Available options:\n{room_list_str}\n\n"
                f"You can reply with something like `Deluxe King Room` or `Standard Queen Room`."
            )
        else:
            # Store room type in a consistent place for execution agent
            booking_state["selected_room_type"] = chosen
            booking_state["room_type"] = chosen

            # Also keep in GuestInfo if that field exists
            if hasattr(guest_info, "room_type"):
                guest_info.room_type = chosen
                booking_state["guest_info"] = guest_info.model_dump()

            booking_state["stage"] = "collecting"

            # Now show your wording: "Would you like to proceed with the booking? I'll need…"
            proceed_line = (
                "Would you like to proceed with the booking? "
                "I'll need to collect some information from you to complete the reservation."
            )

            response = (
                f"Excellent choice! I'll book a <b>{chosen}</b> at <b>{hotel_name}</b>.\n\n"
                f"{proceed_line}\n\n"
                f"Please share:\n"
                f"1. <b>Check-in date</b> (e.g., December 15, 2025)\n"
                f"2. <b>Check-out date</b> (e.g., December 18, 2025)\n"
                f"3. <b>Number of guests</b>\n\n"
                f"You can send this all in one message, like:\n"
                f"`Check-in December 15, check-out December 18, 2 guests`."
            )

    # -------------------- STAGE: COLLECTING -------------------- #
    elif stage == "collecting":
        # Handle cancellation as before
        if any(
            word in user_message.lower()
            for word in ["cancel", "nevermind", "no thanks", "stop"]
        ):
            response = prompts.format("booking_collection.cancelled")
            booking_state["stage"] = "cancelled"
            state["route"] = "end"

        else:
            # Any message here is treated as details (no yes/no confirmation step in this stage)
            guest_info, extracted_fields = extract_guest_info_from_message(
                user_message, guest_info
            )
            booking_state["guest_info"] = guest_info.model_dump()

            if _booking_details_complete(guest_info):
                # Validate booking dates
                is_valid, error_message = _validate_booking_dates(guest_info)
                
                if not is_valid:
                    # Clear the invalid dates so user can re-enter them
                    guest_info.check_in_date = None
                    guest_info.check_out_date = None
                    booking_state["guest_info"] = guest_info.model_dump()
                    
                    response = error_message + "\n\n" + "Please provide new check-in and check-out dates."
                else:
                    # Dates are valid, proceed to confirmation
                    room_type = (
                        booking_state.get("room_type")
                        or booking_state.get("selected_room_type")
                        or "selected room type"
                    )
                    check_in_str = guest_info.check_in_date.strftime("%B %d, %Y")
                    check_out_str = guest_info.check_out_date.strftime("%B %d, %Y")
                    guests_str = guest_info.num_guests
                    
                    response = (
                        f"Let me confirm your booking details:\n\n"
                        f"- <b>Hotel:</b> {hotel_name}\n"
                        f"- <b>Room type:</b> {room_type}\n"
                        f"- <b>Check-in:</b> {check_in_str}\n"
                        f"- <b>Check-out:</b> {check_out_str}\n"
                        f"- <b>Guests:</b> {guests_str}\n\n"
                        f"Is all of this information correct? Please reply with <b>yes</b> or <b>no</b>."
                    )
                    booking_state["stage"] = "confirming"
                    booking_state["confirmation_pending"] = True
            else:
                # Ask only for missing core fields (dates/guests)
                missing = _missing_core_fields(guest_info)

                if extracted_fields:
                    response = (
                        f"Thank you! I've noted: {', '.join(extracted_fields)}.\n\n"
                    )
                else:
                    response = ""

                response += prompts.format(
                    "booking_collection.request_missing",
                    missing_fields=", ".join(missing),
                )

    # -------------------- STAGE: CONFIRMING -------------------- #
    elif stage == "confirming":
        text = user_message.strip().lower()

        if any(word in text for word in ["yes", "confirm", "correct", "proceed", "book"]):
            # Move to executing – booking_execution_agent will handle the rest
            response = prompts.format("booking_collection.processing")
            booking_state["stage"] = "executing"
            state["route"] = "booking_execution"
        elif any(word in text for word in ["no", "wrong", "change", "edit"]):
            response = prompts.format("booking_collection.edit_info")
            booking_state["stage"] = "collecting"
            booking_state["confirmation_pending"] = False
        else:
            response = "Please confirm if the information is correct by saying 'yes' or 'no'."

    else:
        # Fallback – reset if unknown stage
        response = (
            "Let's restart your booking. Which hotel and room type would you like to book?"
        )
        booking_state["stage"] = "initial"
        booking_state["confirmation_pending"] = False

    # Update state
    state["booking_conversation"] = booking_state

    # Add response to messages
    msgs = state.get("messages", [])
    msgs.append({"role": "assistant", "content": response})
    state["messages"] = msgs

    return state