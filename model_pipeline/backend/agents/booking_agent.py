"""
Booking Agent
=============

Handles hotel booking and reservation intents.
"""

import json

from .state import HotelIQState
from .config import bookings_log, BOOKINGS_PATH
from .utils import pick_hotel_for_booking, get_history
from .prompt_loader import get_prompts


def booking_node(state: HotelIQState) -> HotelIQState:
    """
    Booking Agent: Handles hotel booking/reservation intent.
    
    Features:
    - Uses hotel_id from state for direct booking
    - Creates booking records with full hotel context
    - Provides next steps for booking completion
    """
    thread_id = state.get("thread_id", "unknown_thread")
    hotel_id = state.get("hotel_id", "")
    user_message = state["messages"][-1]["content"]
    
    print(f"📝 Booking Agent processing for hotel_id: {hotel_id}")
    
    # Get hotel information from metadata
    hotel_name = "Unknown Hotel"
    star = ""
    
    if "metadata" in state and state["metadata"].get("hotel_info"):
        hotel_info = state["metadata"]["hotel_info"]
        hotel_name = hotel_info.get("name", "Unknown Hotel")
        star = hotel_info.get("star_rating", "")
    elif "metadata" in state and state["metadata"].get("hotel_name"):
        hotel_name = state["metadata"]["hotel_name"]
    
    prompts = get_prompts()

    if not hotel_id:
        answer = "I apologize, but I need a valid hotel ID to process your booking. Please try again."
    else:
        # Store a simple booking stub in our in-memory 'database'
        booking_record = {
            "thread_id": thread_id,
            "hotel_id": hotel_id,
            "hotel_name": hotel_name,
            "star_rating": star,
            "raw_request": user_message,
        }
        bookings_log.append(booking_record)

        # Persist to JSON (fake DB)
        try:
            with open(BOOKINGS_PATH, "w", encoding="utf-8") as f:
                json.dump(bookings_log, f, indent=2)
        except Exception as e:
            print("⚠️ Failed to write bookings JSON:", e)

        answer = prompts.format("booking_agent.booking_success", hotel_name=hotel_name)
        print(f"✅ Booking created for {hotel_name} (ID: {hotel_id})")

    # Update state and history
    msgs = state.get("messages", [])
    msgs.append({"role": "assistant", "content": answer})
    state["messages"] = msgs

    # Also store in chat history
    history_obj = get_history(f"compare_{thread_id}")
    history_obj.add_user_message(user_message)
    history_obj.add_ai_message(answer)

    state["route"] = "end"
    return state

