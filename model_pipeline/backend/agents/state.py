"""
State Definition
================

Contains the state TypedDict for the LangGraph agent system.
"""

from typing import Any, Dict, List, Literal, TypedDict


class HotelIQState(TypedDict, total=False):
    """State for the HotelIQ agent graph."""
    
    messages: List[Dict[str, str]]   # {"role": "user"/"assistant", "content": str}
    user_id: str
    thread_id: str
    hotel_id: str  # Current hotel ID for context-specific queries
    intent: Literal["comparison", "booking", "review"]
    route: Literal["comparison", "booking", "review", "metadata", "end"]
    
    # Metadata tracking for hotel references
    metadata: Dict[str, Any]  # Contains: hotels_mentioned (list by recency), resolved_query, original_query

