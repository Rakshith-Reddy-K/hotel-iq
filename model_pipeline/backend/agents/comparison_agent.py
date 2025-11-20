"""
Comparison Agent
================

Handles hotel information retrieval and similar hotel search queries.
"""

from typing import Dict, List
from langchain_core.documents import Document

from .state import HotelIQState
from .config import last_suggestions, conversation_context
from .pinecone_retrieval import find_similar_hotels  # Only for similar hotel search
from .utils import (
    comparison_chain, get_history, get_limited_history_text,
    build_context_text, detect_comparison_intent, resolve_query_with_context
)


def detect_similar_hotel_intent(query: str) -> bool:
    """
    Detect if user wants to find similar hotels.
    
    Args:
        query: User query
        
    Returns:
        True if similar hotel search is requested
    """
    similar_keywords = ["similar", "like this", "alternatives", "other hotels", "comparable", "find hotels like"]
    query_lower = query.lower()
    return any(keyword in query_lower for keyword in similar_keywords)




def format_similar_hotels_response(similar_hotels: List[Document]) -> str:
    """
    Format similar hotels for display to user with links and information.
    
    Args:
        similar_hotels: List of similar hotel documents
        
    Returns:
        Formatted string with hotel information
    """
    if not similar_hotels:
        return "I couldn't find any similar hotels at the moment. Please try a different search."
    
    response = "Here are some similar hotels you might be interested in:\n\n"
    
    for i, hotel in enumerate(similar_hotels, 1):
        # Get hotel information with fallbacks
        name = hotel.metadata.get("hotel_name") or hotel.metadata.get("official_name") or hotel.metadata.get("name") or "Unknown Hotel"
        hotel_id = hotel.metadata.get("hotel_id", "N/A")
        star_rating = hotel.metadata.get("star_rating", "N/A")
        address = hotel.metadata.get("address", "Address not available")
        city = hotel.metadata.get("city", "")
        state = hotel.metadata.get("state", "")
        overall_rating = hotel.metadata.get("overall_rating", "N/A")
        total_reviews = hotel.metadata.get("total_reviews", "")
        
        # Format location
        location = address
        if city and state:
            location += f", {city}, {state}"
        elif city:
            location += f", {city}"
        
        # Format star rating display
        if star_rating != "N/A" and star_rating:
            try:
                stars = int(float(star_rating))
                star_display = f"{'⭐' * stars}"
            except (ValueError, TypeError):
                star_display = str(star_rating)
        else:
            star_display = "Rating not available"
        
        # Truncate description
        description = hotel.page_content[:200]
        if len(hotel.page_content) > 200:
            description += "..."
        
        # Build hotel entry
        response += f"**{i}. {name}** ({star_display})\n"
        response += f"📍 Location: {location}\n"
        
        if overall_rating != "N/A":
            response += f"⭐ Guest Rating: {overall_rating}/5"
            if total_reviews:
                response += f" ({total_reviews} reviews)"
            response += "\n"
        
        response += f"📝 {description}\n"
        response += f"🔗 [View this hotel](#hotel/{hotel_id})\n\n"
    
    return response


def comparison_node(state: HotelIQState) -> HotelIQState:
    """
    Comparison Agent: Handles hotel information retrieval and similar hotel search.
    """
    thread_id = state.get("thread_id", "unknown_thread")
    hotel_id = state.get("hotel_id", "")
    user_message = state["messages"][-1]["content"]
    
    # Track agent execution
    from langfuse.decorators import langfuse_context
    langfuse_context.update_current_observation(
        name="comparison_agent",
        input={"query": user_message, "hotel_id": hotel_id},
        metadata={"agent": "comparison_agent", "thread_id": thread_id}
    )
    
    print(f"🔍 Comparison Agent processing query for hotel_id: {hotel_id}")
    
    # Use resolved query from metadata agent if available
    if "metadata" in state and "resolved_query" in state["metadata"]:
        query_for_retrieval = state["metadata"]["resolved_query"]
        print(f"📌 Using resolved query: '{query_for_retrieval}'")
    else:
        query_for_retrieval = user_message
    
    # Check if user wants similar hotels
    wants_similar = detect_similar_hotel_intent(query_for_retrieval)

    # Handle similar hotel search
    if wants_similar:
        # Use Pinecone to find similar hotels
        try:
            similar_hotels = find_similar_hotels(hotel_id, top_k=3, exclude_current=True)
            answer = format_similar_hotels_response(similar_hotels)
        except Exception as e:
            print(f"⚠️ Error finding similar hotels: {e}")
            answer = "I apologize, but I encountered an error while searching for similar hotels. Please try again."
    else:
        # Regular information query about the current hotel
        # Use hotel info from metadata (loaded from CSV)
        try:
            hotel_info = state.get("metadata", {}).get("hotel_info")
            hotel_name = state.get("metadata", {}).get("hotel_name", "this hotel")
            
            if hotel_info:
                # Build context from CSV data
                context_parts = []
                
                context_parts.append(f"Hotel Name: {hotel_info.get('hotel_name', 'N/A')}")
                context_parts.append(f"Star Rating: {hotel_info.get('star_rating', 'N/A')}")
                context_parts.append(f"Address: {hotel_info.get('address', 'N/A')}, {hotel_info.get('city', '')}, {hotel_info.get('state', '')} {hotel_info.get('zip_code', '')}")
                
                if hotel_info.get('phone'):
                    context_parts.append(f"Phone: {hotel_info.get('phone')}")
                if hotel_info.get('website'):
                    context_parts.append(f"Website: {hotel_info.get('website')}")
                if hotel_info.get('total_rooms'):
                    context_parts.append(f"Total Rooms: {hotel_info.get('total_rooms')}")
                if hotel_info.get('overall_rating'):
                    context_parts.append(f"Overall Rating: {hotel_info.get('overall_rating')}")
                
                if hotel_info.get('description'):
                    context_parts.append(f"\nDescription: {hotel_info.get('description')}")
                
                if hotel_info.get('additional_info'):
                    context_parts.append(f"\nAdditional Information: {hotel_info.get('additional_info')}")
                
                context_text = "\n".join(context_parts)
                
                print(f"📊 Using CSV data for hotel_id {hotel_id}")
                
                # Generate response using LLM with CSV context
                # Track LLM call
                from ..utils.langfuse_tracker import track_llm_call
                
                llm_response = comparison_chain.invoke(
                    {"history": history_text, "context": context_text, "question": user_message}
                )
                
                track_llm_call(
                    prompt=f"Context: {context_text[:500]}\nQuestion: {user_message}",
                    model="claude-sonnet-3-5",
                    response=llm_response,
                    usage={
                        "prompt_tokens": len(context_text.split()) // 0.75 + len(user_message.split()) // 0.75,
                        "completion_tokens": len(str(llm_response).split()) // 0.75,
                        "total_tokens": (len(context_text.split()) + len(user_message.split()) + len(str(llm_response).split())) // 0.75
                    },
                    metadata={"agent": "comparison_agent", "purpose": "hotel_info"}
                )
                
                answer = llm_response
        except Exception as e:
            print(f"⚠️ Error: {e}")
            answer = "I apologize, but I encountered an error. Please try again."
    
    # Track output
    langfuse_context.update_current_observation(
        output={"response": answer[:500]}
    )
    
    # Update state and history
    msgs = state.get("messages", [])
    msgs.append({"role": "assistant", "content": answer})
    state["messages"] = msgs

    history_obj.add_user_message(user_message)
    history_obj.add_ai_message(answer)

    state["route"] = "end"
    return state

