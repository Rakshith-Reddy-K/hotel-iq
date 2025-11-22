"""
Input Validation and Sanitization
==================================

Pydantic models and sanitization functions to prevent prompt injection
and validate user inputs.
"""

import re
from typing import Optional
from pydantic import BaseModel, Field, field_validator


from logger_config import get_logger

logger = get_logger(__name__)

# ======================================================
# PYDANTIC MODELS
# ======================================================

class UserMessageInput(BaseModel):
    """Validated user message input."""
    
    message: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="User message content"
    )
    user_id: str = Field(
        ...,
        min_length=1,
        max_length=100,
        pattern=r'^[a-zA-Z0-9_-]+$',
        description="User identifier (alphanumeric, underscore, hyphen only)"
    )
    hotel_id: str = Field(
        ...,
        min_length=1,
        max_length=50,
        description="Hotel identifier"
    )
    thread_id: Optional[str] = Field(
        None,
        max_length=200,
        pattern=r'^[a-zA-Z0-9_-]+$',
        description="Thread identifier (alphanumeric, underscore, hyphen only)"
    )
    
    @field_validator('message')
    @classmethod
    def sanitize_message(cls, v: str) -> str:
        """Sanitize user message to prevent prompt injection."""
        return sanitize_user_input(v)
    
    @field_validator('hotel_id')
    @classmethod
    def validate_hotel_id(cls, v: str) -> str:
        """Validate hotel_id format."""
        # Remove any potentially dangerous characters
        sanitized = re.sub(r'[^\w-]', '', v)
        if not sanitized:
            raise ValueError("hotel_id must contain at least one alphanumeric character")
        return sanitized


# ======================================================
# SANITIZATION FUNCTIONS
# ======================================================

def sanitize_user_input(text: str, max_length: int = 2000) -> str:
    """
    Sanitize user input to prevent prompt injection and malformed data.
    
    Args:
        text: User input text
        max_length: Maximum allowed length
        
    Returns:
        Sanitized text
    """
    if not text:
        return text
    
    # Truncate to max length
    text = text[:max_length]
    
    # Strip excessive whitespace
    text = ' '.join(text.split())
    
    # Detect and warn about potential prompt injection patterns
    injection_patterns = [
        r'ignore\s+(previous|all)\s+instructions?',
        r'system\s*:',
        r'<\s*system\s*>',
        r'you\s+are\s+now',
        r'forget\s+(everything|all)',
        r'disregard\s+(previous|all)',
    ]
    
    text_lower = text.lower()
    for pattern in injection_patterns:
        if re.search(pattern, text_lower):
            logger.warning("Potential prompt injection detected", pattern=pattern)
            # Optionally, you could reject the input entirely or sanitize it
            # For now, we'll just log it and continue
    
    # Remove control characters (except newlines and tabs)
    text = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', text)
    
    return text


def validate_thread_id(thread_id: Optional[str]) -> Optional[str]:
    """
    Validate and sanitize thread_id.
    
    Args:
        thread_id: Thread identifier
        
    Returns:
        Validated thread_id or None
    """
    if not thread_id:
        return None
    
    # Only allow alphanumeric, underscore, and hyphen
    sanitized = re.sub(r'[^\w-]', '', thread_id)
    
    # Limit length
    sanitized = sanitized[:200]
    
    return sanitized if sanitized else None


def validate_hotel_id(hotel_id: str) -> str:
    """
    Validate and sanitize hotel_id.
    
    Args:
        hotel_id: Hotel identifier
        
    Returns:
        Validated hotel_id
        
    Raises:
        ValueError: If hotel_id is invalid
    """
    if not hotel_id:
        raise ValueError("hotel_id is required")
    
    # Remove any potentially dangerous characters
    sanitized = re.sub(r'[^\w-]', '', hotel_id)
    
    # Limit length
    sanitized = sanitized[:50]
    
    if not sanitized:
        raise ValueError("hotel_id must contain at least one alphanumeric character")
    
    return sanitized


# ======================================================
# VALIDATION HELPERS
# ======================================================

def validate_chat_request(
    message: str,
    user_id: str,
    hotel_id: str,
    thread_id: Optional[str] = None
) -> dict:
    """
    Validate and sanitize a chat request.
    
    Args:
        message: User message
        user_id: User identifier
        hotel_id: Hotel identifier
        thread_id: Optional thread identifier
        
    Returns:
        Dictionary with validated and sanitized inputs
        
    Raises:
        ValueError: If validation fails
    """
    # Use Pydantic model for validation
    validated = UserMessageInput(
        message=message,
        user_id=user_id,
        hotel_id=hotel_id,
        thread_id=thread_id
    )
    
    return {
        "message": validated.message,
        "user_id": validated.user_id,
        "hotel_id": validated.hotel_id,
        "thread_id": validated.thread_id
    }
