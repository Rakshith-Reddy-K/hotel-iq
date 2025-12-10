"""
Booking State Management
========================

State machine for tracking booking conversation progress.
"""

from typing import TypedDict, Literal, Optional, Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field, field_validator
import re

class GuestInfo(BaseModel):
    """Validated guest information."""
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None
    check_in_date: Optional[datetime] = None
    check_out_date: Optional[datetime] = None
    num_guests: Optional[int] = None
    
    @field_validator('email')
    @classmethod
    def validate_email(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return v
        email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        if not re.match(email_pattern, v):
            raise ValueError("Invalid email format")
        return v.lower()
    
    @field_validator('phone')
    @classmethod
    def validate_phone(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return v
        # Remove all non-digit characters
        digits = re.sub(r'\D', '', v)
        if len(digits) < 10:
            raise ValueError("Phone number must have at least 10 digits")
        return digits
    
    def is_complete(self) -> bool:
        """Check if all required information is collected."""
        return all([
            self.first_name,
            self.last_name,
            self.email,
            self.check_in_date,
            self.check_out_date,
            self.num_guests
        ])
    
    def missing_fields(self) -> list[str]:
        """Return list of missing required fields."""
        missing = []
        if not self.first_name:
            missing.append("first name")
        if not self.last_name:
            missing.append("last name")
        if not self.email:
            missing.append("email")
        if not self.check_in_date:
            missing.append("check-in date")
        if not self.check_out_date:
            missing.append("check-out date")
        if not self.num_guests:
            missing.append("number of guests")
        return missing


class BookingConversationState(TypedDict, total=False):
    """State for booking conversation flow."""
    stage: Literal["initial", "collecting", "confirming", "executing", "completed", "cancelled"]
    guest_info: Dict[str, Any]  # Will store GuestInfo as dict
    hotel_id: str
    hotel_name: str
    hotel_info: Dict[str, Any]
    confirmation_pending: bool
    last_message: str
    booking_id: Optional[str]