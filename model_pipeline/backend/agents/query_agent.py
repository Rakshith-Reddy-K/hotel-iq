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
from .config import llm, last_suggestions, conversation_context
from .utils import get_history, get_limited_history_text
from .prompt_loader import get_prompts
from ..utils.langfuse_tracker import track_agent, track_llm_call


def extract_first_n_words(text: str, n: int = 60) -> str:
    """Extract first n words from text."""
    words = text.split()
    return " ".join(words[:n]) if len(words) > n else text


def extract_hotels_from_text(text: str, use_llm: bool = True) -> List[str]:
    """
    Extract all hotel names from text (assistant response or user message).
    Returns list of hotel names found.
    
    Args:
        text: Text to extract hotel names from
        use_llm: Whether to use LLM for extraction (more accurate but slower)
    """
    hotel_names = []
    text_lower = text.lower()
    
    # Common hotel brand keywords
    hotel_keywords = [
        "radisson", "aka", "sheraton", "westin", "marriott",
        "hilton", "hyatt", "four seasons", "mandarin", "ritz", "omni",
        "renaissance", "doubletree", "courtyard", "fairmont", "intercontinental",
        "lenox", "eliot", "ramada", "comfort inn", "holiday inn", "best western"
    ]
    
    # Quick check if any hotel keywords exist
    has_hotel_keywords = any(keyword in text_lower for keyword in hotel_keywords)
    
    if has_hotel_keywords and use_llm:
        # Use LLM to extract hotel names
        prompts = get_prompts()
        prompt = prompts.format("query_agent.extract_hotel_names_from_text", text=text)
        
        try:
            response = llm.invoke(prompt).content.strip()
            # Parse response - expecting comma-separated list or one per line
            if response and response.lower() != "none":
                # Try splitting by newline or comma
                if '\n' in response:
                    names = [n.strip().strip('-*•123456789.').strip() for n in response.split('\n')]
                else:
                    names = [n.strip() for n in response.split(',')]
                
                # Filter valid names
                for name in names:
                    name = name.strip('"\'.,!? ')
                    if 3 < len(name) < 100 and not name.lower().startswith("there is no"):
                        hotel_names.append(name)
                        
        except Exception as e:
            print(f"⚠️ LLM hotel extraction failed: {e}")
    
    return hotel_names


def extract_explicit_hotel_name(user_message: str, thread_id: str) -> Optional[str]:
    """
    Extract explicit hotel name from user message.
    Returns the first hotel name if found, None otherwise.
    
    Handles patterns like:
    - "Tell me about Hotel Radisson"
    - "Info about the Marriott"
    - "What about the Westin?"
    """
    hotels = extract_hotels_from_text(user_message, use_llm=True)
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
        # Use the dynamically configured hotel path from config
        from .config import HOTELS_PATH
        
        if not HOTELS_PATH.exists():
            print(f"⚠️ CSV file not found at {HOTELS_PATH}")
            return None
        
        # Read CSV
        df = pd.read_csv(HOTELS_PATH)
        
        # Find hotel by ID
        hotel_row = df[df['hotel_id'] == int(hotel_id)]
        
        if hotel_row.empty:
            print(f"⚠️ Hotel ID {hotel_id} not found in CSV")
            return None
        
        # Extract hotel information
        hotel = hotel_row.iloc[0]
        
        # Helper function to convert pandas/numpy types to native Python types
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
        print(f"⚠️ Error retrieving hotel info from CSV: {e}")
        import traceback
        traceback.print_exc()
        return None


def metadata_agent_node(state: HotelIQState) -> HotelIQState:
    """
    Metadata Agent: Manages hotel context and resolves queries with hotel information.
    """
    thread_id = state.get("thread_id", "unknown_thread")
    hotel_id = state.get("hotel_id", "")
    user_message = state["messages"][-1]["content"]
    
    print(f"🏨 Metadata Agent processing query for hotel_id: {hotel_id}")
    
    # Track this agent execution
    from langfuse.decorators import langfuse_context
    langfuse_context.update_current_observation(
        name="query_agent",
        input={"query": user_message, "hotel_id": hotel_id},
        metadata={"agent": "query_agent", "thread_id": thread_id}
    )
    
    # Initialize conversation context for this thread if not present
    if thread_id not in conversation_context:
        conversation_context[thread_id] = {
            "questions": [],
            "hotel_id": hotel_id,
            "hotel_name": None,
            "hotel_info": None,
            "conversation_pairs": []
        }
    
    context = conversation_context[thread_id]
    
    # Update hotel_id in context if it changed
    if context.get("hotel_id") != hotel_id:
        context["hotel_id"] = hotel_id
        context["hotel_name"] = None
        context["hotel_info"] = None
    
    # Retrieve hotel information if not already cached
    if not context.get("hotel_info"):
        hotel_info = get_hotel_info_by_id(hotel_id)
        if hotel_info:
            context["hotel_info"] = hotel_info
            context["hotel_name"] = hotel_info.get("name", "Unknown Hotel")
            print(f"✅ Retrieved hotel info: {context['hotel_name']}")
        else:
            # Handle case where hotel is not found
            context["hotel_name"] = f"Hotel ID {hotel_id}"
    
    # Initialize metadata if not present
    if "metadata" not in state or not state["metadata"]:
        state["metadata"] = {
            "hotel_id": hotel_id,
            "hotel_name": context.get("hotel_name", ""),
            "hotel_info": context.get("hotel_info"),
            "original_query": user_message,
            "resolved_query": user_message,
            "conversation_history": []
        }
    
    # Get current metadata
    metadata = state["metadata"]
    metadata["original_query"] = user_message
    metadata["hotel_id"] = hotel_id
    metadata["hotel_name"] = context.get("hotel_name", "")
    metadata["hotel_info"] = context.get("hotel_info")
    
    # Track conversation history
    context["questions"].append(user_message)
    
    # Build conversation pairs from message history
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
        
        # Get the most recent assistant response for context resolution
        if assistant_messages:
            previous_assistant_response = assistant_messages[-1].get("content", "")
    
    # Detect contextual references in user query
    has_contextual_reference = any(keyword in user_message.lower() for keyword in [
        "this hotel", "that hotel", "this one", "that one", "it", "compare this", "compare it"
    ])
    
    # Enrich query with hotel context
    resolved_query = user_message
    hotel_name = context.get("hotel_name", "")
    
    # Extract all hotels mentioned in user query
    user_mentioned_hotels = extract_hotels_from_text(user_message, use_llm=True)
    
    # If user has contextual references AND previous assistant response exists
    if has_contextual_reference and previous_assistant_response:
        print(f"🔍 Detected contextual reference in query, analyzing previous response...")
        assistant_hotels = extract_hotels_from_text(previous_assistant_response, use_llm=True)
        
        if assistant_hotels:
            print(f"🏨 Hotels found in previous response: {assistant_hotels}")
            print(f"🏨 Hotels mentioned by user: {user_mentioned_hotels}")
            
            prompts = get_prompts()
            prompt = prompts.format(
                "query_agent.resolve_comparison_query",
                previous_response=previous_assistant_response[:500],
                user_query=user_message
            )
            
            try:
                # Track LLM call
                llm_response = llm.invoke(prompt)
                resolved_query = llm_response.content.strip()
                
                # Track token usage
                track_llm_call(
                    prompt=prompt,
                    model="claude-sonnet-3-5",  # Update with your actual model
                    response=resolved_query,
                    usage={
                        "prompt_tokens": getattr(llm_response, 'usage', {}).get('input_tokens', 0) if hasattr(llm_response, 'usage') else 0,
                        "completion_tokens": getattr(llm_response, 'usage', {}).get('output_tokens', 0) if hasattr(llm_response, 'usage') else 0,
                        "total_tokens": getattr(llm_response, 'usage', {}).get('total_tokens', 0) if hasattr(llm_response, 'usage') else 0
                    },
                    metadata={"agent": "query_agent", "purpose": "resolve_comparison_query"}
                )
                
                resolved_query = resolved_query.strip('"\'')
                print(f"✅ Resolved query: '{user_message}' → '{resolved_query}'")
            except Exception as e:
                print(f"⚠️ Query resolution failed: {e}")
                # Fallback: simple substitution with first hotel from assistant response
                if assistant_hotels:
                    resolved_query = user_message.replace("this hotel", assistant_hotels[0])
                    resolved_query = resolved_query.replace("this one", assistant_hotels[0])
                    resolved_query = resolved_query.replace("it", assistant_hotels[0])
                    print(f"🔄 Fallback resolution: '{user_message}' → '{resolved_query}'")
    
    # Else if query is contextual but no explicit hotel in query, use tracked hotel
    elif hotel_name and hotel_name.lower() not in user_message.lower():
        # Check if query needs hotel context
        contextual_keywords = ["amenities", "facilities", "rooms", "price", "location", "here"]
        if any(keyword in user_message.lower() for keyword in contextual_keywords):
            resolved_query = f"{user_message} (regarding {hotel_name})"
            print(f"🔄 Enriched query: '{user_message}' → '{resolved_query}'")
    
    # Update metadata
    metadata["resolved_query"] = resolved_query
    metadata["conversation_history"] = context["questions"][-10:]  # Keep last 10 questions
    
    state["metadata"] = metadata
    
    # Pass through to supervisor
    state["route"] = "supervisor"
    return state


def rewrite_query_with_llm(user_message: str, hotel_name: str, context: Dict[str, Any], thread_id: str) -> str:
    """
    Use LLM to naturally rewrite a query with the hotel name, using conversation context.
    Now includes assistant responses (first 60 words) for better context understanding.
    """
    # Build conversation summary with both questions AND assistant responses
    conversation_pairs = context.get("conversation_pairs", [])
    
    if conversation_pairs:
        # Use the last 3 conversation turns for context
        recent_pairs = conversation_pairs[-3:]
        conversation_summary = ""
        for i, (user_q, assistant_snippet) in enumerate(recent_pairs, 1):
            conversation_summary += f"Turn {i}:\n"
            conversation_summary += f"  User: {user_q}\n"
            conversation_summary += f"  Assistant: {assistant_snippet}...\n\n"
    else:
        # Fallback to old method if no pairs available
        recent_questions = context["questions"][-5:] if len(context["questions"]) > 1 else context["questions"]
        conversation_summary = "\n".join([f"- {q}" for q in recent_questions[:-1]])  # Exclude current question
    
    # Build hotels discussed list
    hotels_discussed = "\n".join([f"{i+1}. {h}" for i, h in enumerate(context["hotels_discussed"])])
    
    prompts = get_prompts()
    prompt = prompts.format(
        "query_agent.rewrite_query_with_context",
        conversation_summary=conversation_summary,
        hotels_discussed=hotels_discussed,
        user_message=user_message,
        hotel_name=hotel_name
    )
    
    try:
        resolved = llm.invoke(prompt).content.strip()
        # Remove quotes if LLM added them
        resolved = resolved.strip('"\'')
        # Validation: ensure hotel name is in the result
        if hotel_name.lower() in resolved.lower():
            return resolved
    except Exception as e:
        print(f"⚠️ LLM rewrite failed: {e}")
    
    # Fallback: simple concatenation
    if "how are the reviews" in user_message.lower():
        return f"How are the reviews for {hotel_name}?"
    elif "what about" in user_message.lower():
        return user_message.replace("what about", f"what about {hotel_name}'s")
    else:
        return f"{user_message} for {hotel_name}"


def resolve_hotel_reference(user_message: str, suggestions: List[Dict[str, str]], thread_id: str) -> tuple[str, Optional[Dict[str, str]]]:
    """
    Enhanced reference resolution that handles:
    - Ordinal references: "the first one", "second hotel", "third one"
    - Numeric references: "hotel 1", "number 2", "#3"
    - Positional: "the last one", "the previous one"
    - Pronouns: "this hotel", "that one", "it", "its"
    
    Returns: (resolved_query, referenced_hotel_dict or None)
    """
    text = user_message.lower()
    
    if not suggestions:
        return user_message, None
    
    # Patterns for different types of references
    ordinal_map = {
        "first": 0, "1st": 0, "one": 0,
        "second": 1, "2nd": 1, "two": 1,
        "third": 2, "3rd": 2, "three": 2,
        "fourth": 3, "4th": 3, "four": 3,
        "fifth": 4, "5th": 4, "five": 4,
    }
    
    referenced_hotel = None
    hotel_index = None
    
    # Check for numeric references: "hotel 1", "number 2", "#3"
    numeric_match = re.search(r'(?:hotel|number|#)\s*(\d+)', text)
    if numeric_match:
        num = int(numeric_match.group(1))
        if 1 <= num <= len(suggestions):
            hotel_index = num - 1  # Convert to 0-based index
    
    # Check for ordinal references: "first one", "second hotel"
    if hotel_index is None:
        for ordinal, idx in ordinal_map.items():
            if ordinal in text and idx < len(suggestions):
                hotel_index = idx
                break
    
    # Check for positional references
    if hotel_index is None:
        if "last" in text or "latest" in text:
            hotel_index = len(suggestions) - 1
        elif "previous" in text and len(suggestions) > 1:
            hotel_index = len(suggestions) - 2
    
    # Check for pronouns/demonstratives (default to most recent)
    contextual_refs = [
        "this hotel", "that hotel", "the hotel", "this one", "that one",
        "it", "its", "their", "they", "there"
    ]
    has_pronoun_reference = any(ref in text for ref in contextual_refs)
    
    if hotel_index is None and has_pronoun_reference:
        hotel_index = len(suggestions) - 1  # Most recent
    
    # If we found a reference, resolve it
    if hotel_index is not None and 0 <= hotel_index < len(suggestions):
        referenced_hotel = suggestions[hotel_index]
        hotel_name = referenced_hotel.get("name", "")
        
        if hotel_name:
            # Use LLM to rewrite query naturally
            history_obj = get_history(f"compare_{thread_id}")
            history_text = get_limited_history_text(history_obj)
            
            # Build hotels list
            hotels_list = "\n".join([f"{i+1}. {h.get('name', 'Unknown')}" for i, h in enumerate(suggestions)])
            
            prompts = get_prompts()
            rewrite_prompt = prompts.format(
                "query_agent.rewrite_query_with_reference",
                history_text=history_text,
                hotels_list=hotels_list,
                user_message=user_message,
                hotel_name=hotel_name
            )
            
            try:
                resolved = llm.invoke(rewrite_prompt).content.strip()
                # Validation: ensure hotel name is in the result
                if hotel_name.lower() in resolved.lower():
                    return resolved, referenced_hotel
            except Exception as e:
                print(f"⚠️ LLM rewrite failed: {e}")
            
            # Fallback: simple replacement
            resolved = user_message
            for ref_pattern in ["the first one", "first one", "the second one", "second one", 
                               "this hotel", "that hotel", "this one", "that one", "it"]:
                if ref_pattern in text:
                    resolved = resolved.replace(ref_pattern, hotel_name)
                    break
            
            return resolved, referenced_hotel
    
    # No reference found, return original
    return user_message, None


@track_agent("query_agent")
def process_query(query: str, **kwargs):
    # When making LLM call, wrap it:
    # response = llm.generate(prompt)
    
    # Track the LLM call
    response = track_llm_call(
        prompt=prompt,
        model=kwargs.get('model', 'gpt-4'),  # or your model name
        response=llm_response,
        usage={
            "prompt_tokens": llm_response.usage.prompt_tokens if hasattr(llm_response, 'usage') else 0,
            "completion_tokens": llm_response.usage.completion_tokens if hasattr(llm_response, 'usage') else 0,
            "total_tokens": llm_response.usage.total_tokens if hasattr(llm_response, 'usage') else 0
        },
        metadata={"agent": "query_agent"}
    )
    
    # ...existing code...
    return response

