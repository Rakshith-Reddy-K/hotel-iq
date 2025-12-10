"""
Booking Agent
=============

Handles hotel booking and reservation intents.

New behavior:
- Multi-turn flow
- Confirms hotel choice
- Collects booking details in any order
- Writes booking record to JSON
- Sends confirmation email via SMTP
"""

import json
import os
from datetime import datetime
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

from .state import HotelIQState
from .config import bookings_log, BOOKINGS_PATH, llm
from .utils import get_history
from .prompt_loader import get_prompts
from logger_config import get_logger

from .langfuse_tracking import track_agent

logger = get_logger(__name__)

# --------------------------------------------------------------------------------------
# In-memory booking session store (per thread)
# --------------------------------------------------------------------------------------

# booking_sessions[thread_id] = {
#   "stage": "awaiting_confirmation" | "collecting_details" | "completed",
#   "hotel_id": str,
#   "hotel_name": str,
#   "star_rating": str,
#   "details": {...}
# }
booking_sessions: Dict[str, Dict[str, Any]] = {}


def get_booking_session(thread_id: str) -> Optional[Dict[str, Any]]:
    return booking_sessions.get(thread_id)


def set_booking_session(thread_id: str, session: Dict[str, Any]) -> None:
    booking_sessions[thread_id] = session


def clear_booking_session(thread_id: str) -> None:
    booking_sessions.pop(thread_id, None)


# --------------------------------------------------------------------------------------
# Structured extraction model for booking details
# --------------------------------------------------------------------------------------


class BookingDetails(BaseModel):
    first_name: str = Field("", description="Guest first name")
    last_name: str = Field("", description="Guest last name")
    email: str = Field(
        "",
        description="Guest email address for confirmation (plain string, no extra text)",
    )
    check_in_date: str = Field(
        "",
        description="Check-in date in ISO format YYYY-MM-DD if you can infer it",
    )
    check_out_date: str = Field(
        "",
        description="Check-out date in ISO format YYYY-MM-DD if you can infer it",
    )
    nights: Optional[int] = Field(
        None,
        description="Number of nights if provided (integer)",
    )
    num_guests: Optional[int] = Field(
        None,
        description="Number of guests if provided (integer)",
    )


booking_parser = JsonOutputParser(pydantic_object=BookingDetails)

booking_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are a hotel booking extraction assistant.

Task:
- Read the guest's message.
- Extract booking details into JSON using the given schema.
- Accept dates in ANY human format (e.g. "15 Dec", "12/15/2025", "15-12-25", "Dec 15th")
- Convert dates to ISO format YYYY-MM-DD whenever possible.
- If you are not sure about a value, leave it empty or null instead of guessing.

Return ONLY valid JSON matching the schema.
""",
        ),
        ("user", "{message}\n\n{format_instructions}"),
    ]
)

booking_chain = booking_prompt | llm | booking_parser

# --------------------------------------------------------------------------------------
# Helper functions
# --------------------------------------------------------------------------------------


def _get_hotel_info_from_state(state: HotelIQState) -> Dict[str, str]:
    """Extract hotel_id, hotel_name, star rating from state + metadata."""
    hotel_id = state.get("hotel_id", "") or ""
    hotel_name = "Unknown Hotel"
    star = ""

    if "metadata" in state and state["metadata"].get("hotel_info"):
        hotel_info = state["metadata"]["hotel_info"]
        hotel_name = hotel_info.get("hotel_name", hotel_info.get("name", "Unknown Hotel"))
        star = hotel_info.get("star_rating", "")
    elif "metadata" in state and state["metadata"].get("hotel_name"):
        hotel_name = state["metadata"]["hotel_name"]

    return {"hotel_id": hotel_id, "hotel_name": hotel_name, "star_rating": star}


def _render_prompt_fallback(path: str, default: str, **kwargs) -> str:
    """Use prompts.yaml if available, otherwise fall back to default text."""
    prompts = get_prompts()
    tmpl = prompts.get(path)
    if not tmpl:
        try:
            return default.format(**kwargs)
        except Exception:
            return default
    try:
        return tmpl.format(**kwargs)
    except Exception:
        return default.format(**kwargs)


def _is_affirmative(text: str) -> bool:
    t = text.lower().strip()
    affirm_words = [
        "yes",
        "yeah",
        "yep",
        "yup",
        "please book",
        "go ahead",
        "confirm",
        "sure",
        "of course",
        "ok book",
        "book it",
        "proceed",
    ]
    return any(word in t for word in affirm_words)


def _is_negative(text: str) -> bool:
    t = text.lower().strip()
    neg_words = [
        "no",
        "not now",
        "cancel",
        "leave it",
        "maybe later",
        "don't book",
        "do not book",
    ]
    return any(word in t for word in neg_words)


def _merge_details(existing: Dict[str, Any], new: BookingDetails) -> Dict[str, Any]:
    """Merge newly parsed details into existing session details."""
    data = existing.copy()
    for field in [
        "first_name",
        "last_name",
        "email",
        "check_in_date",
        "check_out_date",
        "nights",
        "num_guests",
    ]:
        value = getattr(new, field)
        if value not in (None, "", 0):
            # only overwrite if we don't already have a value
            if not data.get(field):
                data[field] = value
    return data


def _missing_fields(details: Dict[str, Any]) -> Dict[str, str]:
    """Return a map of missing required fields -> human-readable labels."""
    required = {
        "first_name": "first name",
        "last_name": "last name",
        "email": "email address",
        "check_in_date": "check-in date",
        "check_out_date": "check-out date",
        "num_guests": "number of guests",
    }
    missing = {}
    for key, label in required.items():
        if not details.get(key):
            missing[key] = label
    return missing


def _format_missing_fields(missing: Dict[str, str]) -> str:
    labels = list(missing.values())
    if not labels:
        return ""
    if len(labels) == 1:
        return labels[0]
    return ", ".join(labels[:-1]) + f" and {labels[-1]}"


def _send_booking_email(to_email: str, hotel_name: str, details: Dict[str, Any]) -> None:
    """Send booking confirmation email via SMTP using environment variables."""
    import smtplib
    from email.message import EmailMessage

    host = os.getenv("SMTP_HOST")
    port = int(os.getenv("SMTP_PORT", "587"))
    username = os.getenv("SMTP_USERNAME")
    password = os.getenv("SMTP_PASSWORD")
    from_email = os.getenv("SMTP_FROM_EMAIL", username or "no-reply@example.com")
    use_tls = os.getenv("SMTP_USE_TLS", "true").lower() in ("1", "true", "yes")

    if not host:
        logger.warning("SMTP not configured; skipping booking email")
        return

    msg = EmailMessage()
    msg["Subject"] = f"Your booking at {hotel_name}"
    msg["From"] = from_email
    msg["To"] = to_email

    check_in = details.get("check_in_date", "")
    check_out = details.get("check_out_date", "")
    nights = details.get("nights")
    num_guests = details.get("num_guests")

    body_lines = [
        f"Dear {details.get('first_name', '')} {details.get('last_name', '')},",
        "",
        f"Thank you for choosing {hotel_name}. Here are your booking details:",
        f"- Check-in date: {check_in}",
        f"- Check-out date: {check_out}",
        f"- Number of nights: {nights if nights else 'N/A'}",
        f"- Number of guests: {num_guests if num_guests is not None else 'N/A'}",
        "",
        "You will need to pay the full amount at check-in at the hotel.",
        "",
        "If any of these details are incorrect, please contact the hotel directly.",
        "",
        "Best regards,",
        "HotelIQ Booking Assistant",
    ]
    msg.set_content("\n".join(body_lines))

    try:
        with smtplib.SMTP(host, port, timeout=20) as server:
            if use_tls:
                server.starttls()
            if username and password:
                server.login(username, password)
            server.send_message(msg)
        logger.info("Booking confirmation email sent", to_email=to_email)
    except Exception as e:
        logger.error("Failed to send booking email", error=str(e))


def _write_booking_record(
    thread_id: str,
    hotel_info: Dict[str, str],
    details: Dict[str, Any],
    raw_request: str,
) -> None:
    """Append booking record to in-memory log + JSON file."""
    record = {
        "thread_id": thread_id,
        "hotel_id": hotel_info["hotel_id"],
        "hotel_name": hotel_info["hotel_name"],
        "star_rating": hotel_info["star_rating"],
        "raw_request": raw_request,
        "first_name": details.get("first_name", ""),
        "last_name": details.get("last_name", ""),
        "email": details.get("email", ""),
        "check_in_date": details.get("check_in_date", ""),
        "check_out_date": details.get("check_out_date", ""),
        "nights": details.get("nights"),
        "num_guests": details.get("num_guests"),
        "created_at": datetime.utcnow().isoformat() + "Z",
    }

    bookings_log.append(record)

    try:
        with open(BOOKINGS_PATH, "w", encoding="utf-8") as f:
            json.dump(bookings_log, f, indent=2)
    except Exception as e:
        logger.error("Failed to write bookings JSON", error=str(e))


# --------------------------------------------------------------------------------------
# MAIN AGENT NODE
# --------------------------------------------------------------------------------------


@track_agent("booking_agent")
async def booking_node(state: HotelIQState) -> HotelIQState:
    """
    Booking Agent: Handles hotel booking/reservation intent.

    Flow:
    1. First time:
       - Confirms the hotel choice.
    2. On user confirmation:
       - Asks for details (name, dates, guests, email).
    3. When user sends details (any order, any date format):
       - Extracts structured info via LLM.
       - Asks only for missing fields.
    4. When all required fields are present:
       - Writes booking record to JSON.
       - Sends confirmation email via SMTP.
       - Informs user that payment is due at check-in.
    """

    thread_id = state.get("thread_id", "unknown_thread")
    user_message = state["messages"][-1]["content"]
    hotel_info = _get_hotel_info_from_state(state)
    hotel_id = hotel_info["hotel_id"]
    hotel_name = hotel_info["hotel_name"]

    history_obj = get_history(f"compare_{thread_id}")

    session = get_booking_session(thread_id)
    if not session:
        # ---------------------------------------------
        # New booking intent: ask for confirmation
        # ---------------------------------------------
        session = {
            "stage": "awaiting_confirmation",
            "hotel_id": hotel_id,
            "hotel_name": hotel_name,
            "star_rating": hotel_info["star_rating"],
            "details": {},
        }
        set_booking_session(thread_id, session)

        logger.info(
            "Booking Collection Agent",
            hotel_name=hotel_name,
            stage="initial",
        )

        answer = _render_prompt_fallback(
            "booking_agent.confirm_choice",
            default="Great choice! {hotel_name} is an excellent option. "
            "Would you like to proceed with booking this hotel? (yes/no)",
            hotel_name=hotel_name,
        )

    else:
        stage = session.get("stage", "awaiting_confirmation")
        details = session.get("details", {}) or {}

        logger.info(
            "Booking Collection Agent",
            hotel_name=hotel_name,
            stage=stage,
        )

        # ---------------------------------------------
        # Stage 1: waiting for yes/no confirmation
        # ---------------------------------------------
        if stage == "awaiting_confirmation":
            if _is_affirmative(user_message):
                session["stage"] = "collecting_details"
                set_booking_session(thread_id, session)
                answer = _render_prompt_fallback(
                    "booking_agent.ask_details",
                    default=(
                        "Perfect! To complete your booking at {hotel_name}, please share:\n"
                        "- First and last name\n"
                        "- Check-in date\n"
                        "- Number of nights or check-out date\n"
                        "- Number of guests\n"
                        "- Email address for confirmation\n\n"
                        "You can provide these in any order, even in one sentence."
                    ),
                    hotel_name=hotel_name,
                )
            elif _is_negative(user_message):
                clear_booking_session(thread_id)
                answer = _render_prompt_fallback(
                    "booking_agent.cancelled",
                    default=(
                        "No problem, I won't book anything right now. "
                        "Let me know if you want to explore other hotels or book later."
                    ),
                )
            else:
                answer = _render_prompt_fallback(
                    "booking_agent.confirm_reprompt",
                    default=(
                        "Just to confirm, would you like to book {hotel_name}? "
                        "Please reply with yes or no."
                    ),
                    hotel_name=hotel_name,
                )

        # ---------------------------------------------
        # Stage 2: collecting booking details
        # ---------------------------------------------
        elif stage == "collecting_details":
            try:
                parsed: BookingDetails = await booking_chain.ainvoke(
                    {
                        "message": user_message,
                        "format_instructions": booking_parser.get_format_instructions(),
                    }
                )
                details = _merge_details(details, parsed)
                session["details"] = details
                set_booking_session(thread_id, session)
            except Exception as e:
                logger.error("Error parsing booking details", error=str(e))
                answer = (
                    "I had trouble understanding all the booking details. "
                    "Please include your full name, check-in date, number of nights or check-out date, "
                    "number of guests, and your email address."
                )
            else:
                missing = _missing_fields(details)

                if missing:
                    missing_text = _format_missing_fields(missing)
                    answer = (
                        f"Got it so far 👍\n\n"
                        f"I still need your {missing_text}. "
                        f"You can just type the missing information in any order."
                    )
                else:
                    # All required details present: finalize booking
                    session["stage"] = "completed"
                    set_booking_session(thread_id, session)

                    _write_booking_record(
                        thread_id=thread_id,
                        hotel_info=hotel_info,
                        details=details,
                        raw_request=user_message,
                    )

                    # Send email (best-effort)
                    if details.get("email"):
                        _send_booking_email(
                            to_email=details["email"],
                            hotel_name=hotel_name,
                            details=details,
                        )

                    clear_booking_session(thread_id)

                    answer = _render_prompt_fallback(
                        "booking_agent.booking_success",
                        default=(
                            "Your booking at {hotel_name} is recorded with the provided details. "
                            "You will need to pay the full amount at check-in at the hotel. "
                            "A confirmation email has been sent to {email} (if email was provided)."
                        ),
                        hotel_name=hotel_name,
                        email=details.get("email", "your email"),
                    )

        else:
            # Completed / unknown stage -> reset
            clear_booking_session(thread_id)
            answer = (
                "Your previous booking session is complete. "
                "If you'd like to book another stay, just say 'book this hotel' again."
            )

    # ----------------------------------------------------------------------------------
    # Update state + history
    # ----------------------------------------------------------------------------------
    msgs = state.get("messages", [])
    msgs.append({"role": "assistant", "content": answer})
    state["messages"] = msgs

    history_obj.add_user_message(user_message)
    history_obj.add_ai_message(answer)

    state["route"] = "end"
    return state

