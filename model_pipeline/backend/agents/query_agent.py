"""
Query/Metadata Agent
====================

Tracks hotels mentioned in conversations and resolves user references.
This agent processes queries before they reach other agents to resolve
contextual references like "this hotel", "the first one", etc.
"""

import re
from typing import Any, Dict, List, Optional

from .state import HotelIQState
from .config import llm
from .utils import get_history, get_limited_history_text
from .prompt_loader import get_prompts
from logger_config import get_logger

logger = get_logger(__name__)


def extract_first_n_words(text: str, n: int = 60) -> str:
    """Extract first n words from text."""
    words = text.split()
    return " ".join(words[:n]) if len(words) > n else text


async def extract_hotels_from_text(text: str, use_llm: bool = True) -> List[str]:
    """
    Extract all hotel names from text (assistant response or user message).
    Returns list of hotel names found.
    
    Args:
        text: Text to extract hotel names from
        use_llm: Whether to use LLM for extraction (more accurate but slower)
    """
    hotel_names = []
    text_lower = text.lower()
    
    hotel_keywords = [
        "radisson", "aka", "sheraton", "westin", "marriott",
        "hilton", "hyatt", "four seasons", "mandarin", "ritz", "omni",
        "renaissance", "doubletree", "courtyard", "fairmont", "intercontinental",
        "lenox", "eliot", "ramada", "comfort inn", "holiday inn", "best western"
    ]
    
    has_hotel_keywords = any(keyword in text_lower for keyword in hotel_keywords)
    
    if has_hotel_keywords and use_llm:
        prompts = get_prompts()
        prompt = prompts.format("query_agent.extract_hotel_names_from_text", text=text)
        
        try:
            response = await llm.ainvoke(prompt)
            response = response.content.strip()
            if response and response.lower() != "none":
                if '\n' in response:
                    names = [n.strip().strip('-*•123456789.').strip() for n in response.split('\n')]
                else:
                    names = [n.strip() for n in response.split(',')]
                
                for name in names:
                    name = name.strip('"\'.,!? ')
                    if 3 < len(name) < 100 and not name.lower().startswith("there is no"):
                        hotel_names.append(name)
                        
        except Exception as e:
            logger.warning("LLM hotel extraction failed", error=str(e))
    
    return hotel_names


async def extract_explicit_hotel_name(user_message: str, thread_id: str) -> Optional[str]:
    """
    Extract explicit hotel name from user message.
    Returns the first hotel name if found, None otherwise.
    
    Handles patterns like:
    - "Tell me about Hotel Radisson"
    - "Info about the Marriott"
    - "What about the Westin?"
    """
    hotels = await extract_hotels_from_text(user_message, use_llm=True)
    return hotels[0] if hotels else None


def get_hotel_info_by_id(hotel_id: str) -> Optional[Dict[str, Any]]:
    """
    Retrieve hotel information from CSV file using hotel_id.
    
    Args:
        hotel_id: Hotel ID to look up
        
    Returns:
        Dictionary with hotel information or None if not found
    """
    import pandas as pd
    from pathlib import Path
    
    try:
        from .config import HOTELS_PATH
        
        if not HOTELS_PATH.exists():
            logger.warning("CSV file not found", path=str(HOTELS_PATH))
            return None
        
        df = pd.read_csv(HOTELS_PATH)
        
        if not hotel_id:
            return None
            
        hotel_row = df[df['hotel_id'] == int(hotel_id)]
        
        if hotel_row.empty:
            logger.warning("Hotel ID not found in CSV", hotel_id=hotel_id)
            return None
        
        hotel = hotel_row.iloc[0]
        
        def to_python_type(value):
            """Convert pandas/numpy types to native Python types for serialization."""
            if pd.isna(value):
                return ""
            if hasattr(value, 'item'):  # numpy types have .item() method
                return value.item()
            return str(value) if value is not None else ""
        
        return {
            "hotel_id": str(hotel_id),
            "name": to_python_type(hotel.get("official_name", "Unknown Hotel")),
            "hotel_name": to_python_type(hotel.get("official_name", "Unknown Hotel")),
            "star_rating": to_python_type(hotel.get("star_rating", "")),
            "description": to_python_type(hotel.get("description", "")),
            "address": to_python_type(hotel.get("address", "")),
            "city": to_python_type(hotel.get("city", "")),
            "state": to_python_type(hotel.get("state", "")),
            "zip_code": to_python_type(hotel.get("zip_code", "")),
            "phone": to_python_type(hotel.get("phone", "")),
            "website": to_python_type(hotel.get("website", "")),
            "total_rooms": to_python_type(hotel.get("total_rooms", "")),
            "overall_rating": to_python_type(hotel.get("overall_rating", "")),
            "additional_info": to_python_type(hotel.get("additional_info", ""))
        }
        
    except Exception as e:
        logger.error("Error retrieving hotel info from CSV", error=str(e))
        import traceback
        traceback.print_exc()
        return None


from .langfuse_tracking import track_agent

@track_agent("metadata_agent")
async def metadata_agent_node(state: HotelIQState) -> HotelIQState:
    """
    Metadata Agent: Manages hotel context and resolves queries with hotel information.
    
    This agent:
    1. Retrieves hotel information using the provided hotel_id
    2. Maintains conversation context for the specific hotel
    3. Tracks conversation history
    4. Enriches queries with hotel context
    5. Passes the enriched query to downstream agents
    """
    thread_id = state.get("thread_id", "unknown_thread")
    hotel_id = state.get("hotel_id", "")
    user_message = state["messages"][-1]["content"]
    
    logger.info("Metadata Agent processing query", hotel_id=hotel_id)
    
    if "conversation_context" not in state or not state["conversation_context"]:
        state["conversation_context"] = {
            "questions": [],
            "hotel_id": hotel_id,
            "hotel_name": None,
            "hotel_info": None,
            "conversation_pairs": []
        }
    
    context = state["conversation_context"]
    
    if context.get("hotel_id") != hotel_id:
        context["hotel_id"] = hotel_id
        context["hotel_name"] = None
        context["hotel_info"] = None
    
    if not context.get("hotel_info"):
        hotel_info = get_hotel_info_by_id(hotel_id)
        if hotel_info:
            context["hotel_info"] = hotel_info
            context["hotel_name"] = hotel_info.get("name", "Unknown Hotel")
            logger.info("Retrieved hotel info", hotel_name=context['hotel_name'])
        else:
            context["hotel_name"] = f"Hotel ID {hotel_id}"
    
    if "metadata" not in state or not state["metadata"]:
        state["metadata"] = {
            "hotel_id": hotel_id,
            "hotel_name": context.get("hotel_name", ""),
            "hotel_info": context.get("hotel_info"),
            "original_query": user_message,
            "resolved_query": user_message,
            "conversation_history": []
        }
    
    metadata = state["metadata"]
    metadata["original_query"] = user_message
    metadata["hotel_id"] = hotel_id
    metadata["hotel_name"] = context.get("hotel_name", "")
    metadata["hotel_info"] = context.get("hotel_info")
    
    context["questions"].append(user_message)
    
    messages = state.get("messages", [])
    previous_assistant_response = ""
    
    if len(messages) > 1:
        user_messages = [msg for msg in messages[:-1] if msg.get("role") == "user"]
        assistant_messages = [msg for msg in messages[:-1] if msg.get("role") == "assistant"]
        
        if len(user_messages) == len(assistant_messages):
            context["conversation_pairs"] = [
                (user_messages[i].get("content", ""), extract_first_n_words(assistant_messages[i].get("content", ""), n=60))
                for i in range(len(user_messages))
            ]
        
        if assistant_messages:
            previous_assistant_response = assistant_messages[-1].get("content", "")
    
    suggestions = state.get("last_suggestions", [])
    hotels_list_str = "None"
    if suggestions:
        hotels_list_str = "\n".join([f"{i+1}. {h.get('name', 'Unknown')}" for i, h in enumerate(suggestions)])
        
    history_obj = get_history(f"compare_{thread_id}")
    history_text = get_limited_history_text(history_obj)
    
    prompts = get_prompts()
    prompt = prompts.format(
        "query_agent.general_query_rewrite",
        history_text=history_text,
        hotels_list=hotels_list_str,
        user_message=user_message
    )
    
    resolved_query = user_message
    try:
        response = await llm.ainvoke(prompt)
        resolved_query = response.content.strip().strip('"\'')
        logger.info("Resolved query", original=user_message, resolved=resolved_query)
    except Exception as e:
        logger.error("Query resolution failed", error=str(e))
        resolved_query = user_message
    
    metadata["resolved_query"] = resolved_query
    metadata["conversation_history"] = context["questions"][-10:]  # Keep last 10 questions
    
    state["metadata"] = metadata
    state["conversation_context"] = context  # Save context back to state
    
    state["route"] = "supervisor"
    return state




