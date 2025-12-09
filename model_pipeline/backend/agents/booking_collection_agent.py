# """
# Booking Collection Agent
# ========================

# Handles natural conversation to collect guest information for booking.
# """

# from typing import Dict, Any
# from datetime import datetime
# import re
# from dateutil import parser as date_parser

# from .state import HotelIQState
# from .booking_state import GuestInfo, BookingConversationState
# from .config import llm
# from .prompt_loader import get_prompts
# from logger_config import get_logger

# logger = get_logger(__name__)


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
#         # Use dateutil parser for flexible parsing
#         parsed_date = date_parser.parse(date_str, fuzzy=True)
        
#         # If year is not specified, assume current year or next year
#         if parsed_date.year == datetime.now().year and parsed_date < datetime.now():
#             parsed_date = parsed_date.replace(year=parsed_date.year + 1)
        
#         return parsed_date
#     except Exception as e:
#         logger.warning("Failed to parse date", date_str=date_str, error=str(e))
#         raise ValueError(f"Could not understand date: {date_str}")


# def extract_guest_info_from_message(message: str, current_info: GuestInfo) -> tuple[GuestInfo, list[str]]:
#     """
#     Extract guest information from user message using LLM.
    
#     Returns:
#         Tuple of (updated GuestInfo, list of extracted field names)
#     """
#     prompts = get_prompts()
    
#     extraction_prompt = prompts.format(
#         "booking_collection.extract_info",
#         message=message,
#         current_info=current_info.model_dump_json()
#     )
    
#     try:
#         response = llm.invoke(extraction_prompt)
#         # Parse LLM response to extract structured data
#         # Expected format: JSON with fields
#         import json
#         extracted_data = json.loads(response.content)
        
#         updated_fields = []
        
#         # Update guest info with extracted data
#         if extracted_data.get('first_name') and not current_info.first_name:
#             current_info.first_name = extracted_data['first_name']
#             updated_fields.append('first_name')
        
#         if extracted_data.get('last_name') and not current_info.last_name:
#             current_info.last_name = extracted_data['last_name']
#             updated_fields.append('last_name')
        
#         if extracted_data.get('email') and not current_info.email:
#             try:
#                 current_info.email = extracted_data['email']
#                 updated_fields.append('email')
#             except ValueError:
#                 pass
        
#         if extracted_data.get('phone') and not current_info.phone:
#             try:
#                 current_info.phone = extracted_data['phone']
#                 updated_fields.append('phone')
#             except ValueError:
#                 pass
        
#         if extracted_data.get('check_in_date') and not current_info.check_in_date:
#             try:
#                 current_info.check_in_date = parse_date_flexible(extracted_data['check_in_date'])
#                 updated_fields.append('check_in_date')
#             except ValueError:
#                 pass
        
#         if extracted_data.get('check_out_date') and not current_info.check_out_date:
#             try:
#                 current_info.check_out_date = parse_date_flexible(extracted_data['check_out_date'])
#                 updated_fields.append('check_out_date')
#             except ValueError:
#                 pass
        
#         if extracted_data.get('num_guests') and not current_info.num_guests:
#             try:
#                 current_info.num_guests = int(extracted_data['num_guests'])
#                 updated_fields.append('num_guests')
#             except ValueError:
#                 pass
        
#         return current_info, updated_fields
        
#     except Exception as e:
#         logger.error("Failed to extract info with LLM", error=str(e))
#         return current_info, []


# async def booking_collection_node(state: HotelIQState) -> HotelIQState:
#     """
#     Booking Collection Agent: Manages conversation to collect guest information.
    
#     Stages:
#     - initial: Confirm booking intent
#     - collecting: Gather guest details
#     - confirming: Final confirmation before execution
#     """
#     user_message = state["messages"][-1]["content"]
#     prompts = get_prompts()
    
#     # Initialize booking state if not exists
#     if "booking_conversation" not in state:
#         state["booking_conversation"] = {
#             "stage": "initial",
#             "guest_info": GuestInfo().model_dump(),
#             "hotel_id": state.get("hotel_id", ""),
#             "hotel_name": state.get("metadata", {}).get("hotel_name", "this hotel"),
#             "hotel_info": state.get("metadata", {}).get("hotel_info", {}),
#             "confirmation_pending": False
#         }
    
#     booking_state = state["booking_conversation"]
#     guest_info = GuestInfo(**booking_state["guest_info"])
#     stage = booking_state["stage"]
#     hotel_name = booking_state["hotel_name"]
    
#     logger.info("Booking Collection Agent", stage=stage, hotel_name=hotel_name)
    
#     # Stage: Initial - Confirm booking intent
#     if stage == "initial":
#         response = prompts.format(
#             "booking_collection.initial_confirmation",
#             hotel_name=hotel_name
#         )
#         booking_state["stage"] = "collecting"
#         booking_state["confirmation_pending"] = True
    
#     # Stage: Collecting - Gather information
#     elif stage == "collecting":
#         # Check if user cancelled
#         if any(word in user_message.lower() for word in ["cancel", "nevermind", "no thanks", "stop"]):
#             response = prompts.format("booking_collection.cancelled")
#             booking_state["stage"] = "cancelled"
#             state["route"] = "end"
        
#         # If confirmation was pending, check response
#         elif booking_state.get("confirmation_pending"):
#             if any(word in user_message.lower() for word in ["yes", "sure", "proceed", "ok", "yeah"]):
#                 booking_state["confirmation_pending"] = False
#                 response = prompts.format("booking_collection.start_collection")
#             else:
#                 response = prompts.format("booking_collection.cancelled")
#                 booking_state["stage"] = "cancelled"
#                 state["route"] = "end"
        
#         # Extract information from message
#         else:
#             guest_info, extracted_fields = extract_guest_info_from_message(user_message, guest_info)
#             booking_state["guest_info"] = guest_info.model_dump()
            
#             # Check if all information is collected
#             if guest_info.is_complete():
#                 # Generate confirmation message
#                 response = prompts.format(
#                     "booking_collection.final_confirmation",
#                     first_name=guest_info.first_name,
#                     last_name=guest_info.last_name,
#                     email=guest_info.email,
#                     check_in=guest_info.check_in_date.strftime("%B %d, %Y"),
#                     check_out=guest_info.check_out_date.strftime("%B %d, %Y"),
#                     num_guests=guest_info.num_guests,
#                     hotel_name=hotel_name
#                 )
#                 booking_state["stage"] = "confirming"
#                 booking_state["confirmation_pending"] = True
#             else:
#                 # Ask for missing information
#                 missing = guest_info.missing_fields()
#                 if extracted_fields:
#                     response = f"Thank you! I've noted: {', '.join(extracted_fields)}.\n\n"
#                 else:
#                     response = ""
                
#                 response += prompts.format(
#                     "booking_collection.request_missing",
#                     missing_fields=", ".join(missing)
#                 )
    
#     # Stage: Confirming - Final confirmation
#     elif stage == "confirming":
#         if any(word in user_message.lower() for word in ["yes", "confirm", "correct", "proceed", "book"]):
#             response = prompts.format("booking_collection.processing")
#             booking_state["stage"] = "executing"
#             # Route to execution agent
#             state["route"] = "booking_execution"
#         elif any(word in user_message.lower() for word in ["no", "wrong", "change", "edit"]):
#             response = prompts.format("booking_collection.edit_info")
#             booking_state["stage"] = "collecting"
#             booking_state["confirmation_pending"] = False
#         else:
#             response = "Please confirm if the information is correct by saying 'yes' or 'no'."
    
#     # Update state
#     state["booking_conversation"] = booking_state
    
#     # Add response to messages
#     msgs = state.get("messages", [])
#     msgs.append({"role": "assistant", "content": response})
#     state["messages"] = msgs
    
#     return state

# """
# Booking Collection Agent
# ========================

# Handles natural conversation to collect guest information for booking.
# """

# from typing import Dict, Any, List
# from datetime import datetime
# import re
# from dateutil import parser as date_parser

# from .state import HotelIQState
# from .booking_state import GuestInfo, BookingConversationState
# from .config import llm
# from .prompt_loader import get_prompts
# from logger_config import get_logger

# logger = get_logger(__name__)


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


# def extract_guest_info_from_message(message: str, current_info: GuestInfo) -> tuple[GuestInfo, list[str]]:
#     """
#     Extract guest information from user message using LLM.

#     NOTE (updated behaviour):
#     - We will only *use* check-in date, check-out date, and num_guests for flow logic.
#     - Name/email/phone may still be extracted but we do NOT ask for them
#       and we don't rely on them to consider the booking "complete".
    
#     Returns:
#         Tuple of (updated GuestInfo, list of extracted field names)
#     """
#     prompts = get_prompts()

#     extraction_prompt = prompts.format(
#         "booking_collection.extract_info",
#         message=message,
#         current_info=current_info.model_dump_json()
#     )

#     try:
#         response = llm.invoke(extraction_prompt)
#         import json
#         extracted_data = json.loads(response.content)

#         updated_fields: list[str] = []

#         # We won't *ask* for these anymore, but we leave the logic for backward compatibility.
#         if extracted_data.get("first_name") and not current_info.first_name:
#             current_info.first_name = extracted_data["first_name"]
#             updated_fields.append("first_name")

#         if extracted_data.get("last_name") and not current_info.last_name:
#             current_info.last_name = extracted_data["last_name"]
#             updated_fields.append("last_name")

#         if extracted_data.get("email") and not current_info.email:
#             try:
#                 current_info.email = extracted_data["email"]
#                 updated_fields.append("email")
#             except ValueError:
#                 pass

#         if extracted_data.get("phone") and not current_info.phone:
#             try:
#                 current_info.phone = extracted_data["phone"]
#                 updated_fields.append("phone")
#             except ValueError:
#                 pass

#         if extracted_data.get("check_in_date") and not current_info.check_in_date:
#             try:
#                 current_info.check_in_date = parse_date_flexible(
#                     extracted_data["check_in_date"]
#                 )
#                 updated_fields.append("check_in_date")
#             except ValueError:
#                 pass

#         if extracted_data.get("check_out_date") and not current_info.check_out_date:
#             try:
#                 current_info.check_out_date = parse_date_flexible(
#                     extracted_data["check_out_date"]
#                 )
#                 updated_fields.append("check_out_date")
#             except ValueError:
#                 pass

#         if extracted_data.get("num_guests") and not current_info.num_guests:
#             try:
#                 current_info.num_guests = int(extracted_data["num_guests"])
#                 updated_fields.append("num_guests")
#             except ValueError:
#                 pass

#         return current_info, updated_fields

#     except Exception as e:
#         logger.error("Failed to extract info with LLM", error=str(e))
#         return current_info, []


# # -------------------- NEW HELPERS FOR ROOM TYPES -------------------- #

# def _get_hotel_info_from_state(state: HotelIQState) -> Dict[str, Any]:
#     """
#     Get hotel info from state/booking_conversation.
#     This reuses your existing metadata/booking structure.
#     """
#     # Prefer booking_conversation if present
#     if "booking_conversation" in state:
#         bc = state["booking_conversation"]
#         if isinstance(bc, dict) and bc.get("hotel_info"):
#             return bc["hotel_info"]

#     # Fallback to metadata set by metadata_agent
#     if "metadata" in state:
#         meta = state["metadata"] or {}
#         if isinstance(meta, dict) and meta.get("hotel_info"):
#             return meta["hotel_info"]

#     return {}


# def _extract_room_types(hotel_info: Dict[str, Any]) -> List[str]:
#     """
#     Derive room types from hotel_info if available,
#     otherwise provide a generic but reasonable list.
    
#     You can customize this to parse a 'room_types' column
#     from your hotels.csv (e.g., pipe-separated string).
#     """
#     # If your hotels.csv has something like "room_types" field
#     if "room_types" in hotel_info and hotel_info["room_types"]:
#         raw = str(hotel_info["room_types"])
#         return [r.strip() for r in re.split(r"[|,]", raw) if r.strip()]

#     # Generic fallback – does NOT break existing bot
#     return [
#         "Standard King Room",
#         "Standard Queen Room",
#         "Deluxe King Room",
#         "Deluxe Queen Room",
#         "Suite",
#     ]


# def _fuzzy_pick_room_type(user_text: str, options: List[str]) -> str | None:
#     """
#     Try to match the user's message to one of the room types.
#     """
#     text = user_text.lower()

#     # Exact substring
#     for opt in options:
#         if opt.lower() in text:
#             return opt

#     # Partial word/keyword match
#     for opt in options:
#         words = [w for w in opt.lower().split() if len(w) > 3]
#         if any(w in text for w in words):
#             return opt

#     return None


# def _booking_details_complete(guest_info: GuestInfo) -> bool:
#     """
#     New notion of 'complete' for this flow:
#     We ONLY require:
#     - check_in_date
#     - check_out_date
#     - num_guests

#     Name/email are now assumed to come from login, not chat.
#     """
#     return bool(
#         guest_info.check_in_date
#         and guest_info.check_out_date
#         and guest_info.num_guests
#     )


# def _missing_core_fields(guest_info: GuestInfo) -> List[str]:
#     """
#     Compute missing fields for the *conversation*.
#     We no longer ask for name/email/phone here.
#     """
#     missing: List[str] = []
#     if not guest_info.check_in_date:
#         missing.append("check-in date")
#     if not guest_info.check_out_date:
#         missing.append("check-out date")
#     if not guest_info.num_guests:
#         missing.append("number of guests")
#     return missing


# # -------------------- MAIN BOOKING COLLECTION NODE -------------------- #

# async def booking_collection_node(state: HotelIQState) -> HotelIQState:
#     """
#     Booking Collection Agent: Manages conversation to collect guest information.

#     UPDATED STAGES:
#     - initial: user just requested booking; greet & show room types
#     - choosing_room_type: user picks a room type
#     - collecting: gather check-in, check-out, num guests
#     - confirming: final confirmation before execution
#     - executing: handled by booking_execution_agent (unchanged)
#     """
#     user_message = state["messages"][-1]["content"]
#     prompts = get_prompts()

#     # Initialize booking state if not exists
#     if "booking_conversation" not in state:
#         state["booking_conversation"] = {
#             "stage": "initial",
#             "guest_info": GuestInfo().model_dump(),
#             "hotel_id": state.get("hotel_id", ""),
#             "hotel_name": state.get("metadata", {}).get("hotel_name", "this hotel"),
#             "hotel_info": state.get("metadata", {}).get("hotel_info", {}),
#             "confirmation_pending": False,
#             "selected_room_type": None,
#             "available_room_types": [],
#         }

#     booking_state = state["booking_conversation"]
#     guest_info = GuestInfo(**booking_state["guest_info"])
#     stage = booking_state["stage"]
#     hotel_name = booking_state["hotel_name"]
#     hotel_info = booking_state.get("hotel_info") or _get_hotel_info_from_state(state)

#     logger.info(
#         "Booking Collection Agent",
#         stage=stage,
#         hotel_name=hotel_name,
#     )

#     # -------------------- STAGE: INITIAL -------------------- #
#     if stage == "initial":
#         # Get room types for this hotel
#         room_types = _extract_room_types(hotel_info)
#         booking_state["available_room_types"] = room_types

#         # Use your existing initial prompt as intro (so tone stays same)
#         intro = prompts.format(
#             "booking_collection.initial_confirmation",
#             hotel_name=hotel_name,
#         )

#         room_list_str = "\n".join(f"- {rt}" for rt in room_types)

#         response = (
#             f"{intro}\n\n"
#             f"Before we proceed, here are the room types available at **{hotel_name}**:\n"
#             f"{room_list_str}\n\n"
#             f"👉 Which room type would you like to book?"
#         )

#         booking_state["stage"] = "choosing_room_type"
#         booking_state["confirmation_pending"] = False  # no yes/no here anymore

#     # -------------------- STAGE: CHOOSING ROOM TYPE -------------------- #
#     elif stage == "choosing_room_type":
#         room_types = booking_state.get("available_room_types") or _extract_room_types(
#             hotel_info
#         )

#         chosen = _fuzzy_pick_room_type(user_message, room_types)
#         if not chosen:
#             room_list_str = "\n".join(f"- {rt}" for rt in room_types)
#             response = (
#                 f"Got it! Just to confirm, which exact room type would you like at **{hotel_name}**?\n\n"
#                 f"Available options:\n{room_list_str}\n\n"
#                 f"You can reply with something like `Deluxe King Room` or `Standard Queen Room`."
#             )
#         else:
#             booking_state["selected_room_type"] = chosen
#             booking_state["stage"] = "collecting"

#             # Now show your wording: "Would you like to proceed with the booking? I'll need…"
#             proceed_line = (
#                 "Would you like to proceed with the booking? "
#                 "I'll need to collect some information from you to complete the reservation."
#             )

#             response = (
#                 f"Excellent choice! I'll book a **{chosen}** at **{hotel_name}**.\n\n"
#                 f"{proceed_line}\n\n"
#                 f"Please share:\n"
#                 f"1. **Check-in date** (e.g., December 15, 2025)\n"
#                 f"2. **Check-out date** (e.g., December 18, 2025)\n"
#                 f"3. **Number of guests**\n\n"
#                 f"You can send this all in one message, like:\n"
#                 f"`Check-in December 15, check-out December 18, 2 guests`."
#             )

#     # -------------------- STAGE: COLLECTING -------------------- #
#     elif stage == "collecting":
#         # Handle cancellation as before
#         if any(
#             word in user_message.lower()
#             for word in ["cancel", "nevermind", "no thanks", "stop"]
#         ):
#             response = prompts.format("booking_collection.cancelled")
#             booking_state["stage"] = "cancelled"
#             state["route"] = "end"

#         else:
#             # We NO LONGER pause on "yes/no" here: any message is treated as details.
#             guest_info, extracted_fields = extract_guest_info_from_message(
#                 user_message, guest_info
#             )
#             booking_state["guest_info"] = guest_info.model_dump()

#             if _booking_details_complete(guest_info):
#                 # Build our own confirmation text (no name/email)
#                 room_type = booking_state.get("selected_room_type") or "selected room type"

#                 check_in_str = guest_info.check_in_date.strftime("%B %d, %Y")
#                 check_out_str = guest_info.check_out_date.strftime("%B %d, %Y")
#                 guests_str = guest_info.num_guests

#                 response = (
#                     f"Let me confirm your booking details:\n\n"
#                     f"- **Hotel:** {hotel_name}\n"
#                     f"- **Room type:** {room_type}\n"
#                     f"- **Check-in:** {check_in_str}\n"
#                     f"- **Check-out:** {check_out_str}\n"
#                     f"- **Guests:** {guests_str}\n\n"
#                     f"Is all of this information correct? Please reply with **yes** or **no**."
#                 )

#                 booking_state["stage"] = "confirming"
#                 booking_state["confirmation_pending"] = True
#             else:
#                 # Ask only for missing core fields (dates/guests)
#                 missing = _missing_core_fields(guest_info)

#                 if extracted_fields:
#                     response = (
#                         f"Thank you! I've noted: {', '.join(extracted_fields)}.\n\n"
#                     )
#                 else:
#                     response = ""

#                 response += prompts.format(
#                     "booking_collection.request_missing",
#                     missing_fields=", ".join(missing),
#                 )

#     # -------------------- STAGE: CONFIRMING -------------------- #
#     elif stage == "confirming":
#         text = user_message.strip().lower()

#         if any(word in text for word in ["yes", "confirm", "correct", "proceed", "book"]):
#             # Move to executing – booking_execution_agent will handle the rest
#             response = prompts.format("booking_collection.processing")
#             booking_state["stage"] = "executing"
#             state["route"] = "booking_execution"  # still safe even with new graph
#         elif any(word in text for word in ["no", "wrong", "change", "edit"]):
#             response = prompts.format("booking_collection.edit_info")
#             booking_state["stage"] = "collecting"
#             booking_state["confirmation_pending"] = False
#         else:
#             response = "Please confirm if the information is correct by saying 'yes' or 'no'."

#     else:
#         # Fallback – reset if unknown stage
#         response = (
#             "Let's restart your booking. Which hotel and room type would you like to book?"
#         )
#         booking_state["stage"] = "initial"
#         booking_state["confirmation_pending"] = False

#     # Update state
#     state["booking_conversation"] = booking_state

#     # Add response to messages
#     msgs = state.get("messages", [])
#     msgs.append({"role": "assistant", "content": response})
#     state["messages"] = msgs

#     return state


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

logger = get_logger(__name__)


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

        # If year is not specified, assume current year or next year
        now = datetime.now()
        if parsed_date.year == now.year and parsed_date < now:
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


def _extract_room_types(hotel_info: Dict[str, Any]) -> List[str]:
    """
    Derive room types from hotel_info if available,
    otherwise provide a generic but reasonable list.
    
    You can customize this to parse a 'room_types' column
    from your hotels.csv (e.g., pipe-separated string).
    """
    # If your hotels.csv has something like "room_types" field
    if "room_types" in hotel_info and hotel_info["room_types"]:
        raw = str(hotel_info["room_types"])
        return [r.strip() for r in re.split(r"[|,]", raw) if r.strip()]

    # Generic fallback – does NOT break existing bot
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
        room_types = _extract_room_types(hotel_info)
        booking_state["available_room_types"] = room_types

        # Use your existing initial prompt as intro (so tone stays same)
        intro = prompts.format(
            "booking_collection.initial_confirmation",
            hotel_name=hotel_name,
        )

        room_list_str = "\n".join(f"- {rt}" for rt in room_types)

        response = (
            f"{intro}\n\n"
            f"Before we proceed, here are the room types available at **{hotel_name}**:\n"
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
                f"Got it! Just to confirm, which exact room type would you like at **{hotel_name}**?\n\n"
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
                f"Excellent choice! I'll book a **{chosen}** at **{hotel_name}**.\n\n"
                f"{proceed_line}\n\n"
                f"Please share:\n"
                f"1. **Check-in date** (e.g., December 15, 2025)\n"
                f"2. **Check-out date** (e.g., December 18, 2025)\n"
                f"3. **Number of guests**\n\n"
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
                # Build our own confirmation text (no name/email)
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
                    f"- **Hotel:** {hotel_name}\n"
                    f"- **Room type:** {room_type}\n"
                    f"- **Check-in:** {check_in_str}\n"
                    f"- **Check-out:** {check_out_str}\n"
                    f"- **Guests:** {guests_str}\n\n"
                    f"Is all of this information correct? Please reply with **yes** or **no**."
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
