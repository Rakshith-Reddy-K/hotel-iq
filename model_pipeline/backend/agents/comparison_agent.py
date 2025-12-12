# """
# Comparison Agent
# ================

# Handles hotel information retrieval and similar hotel search queries.
# """

# from typing import Dict, List
# from langchain_core.documents import Document

# from .state import HotelIQState
# from .pinecone_retrieval import find_similar_hotels 
# from .utils import (
#     comparison_chain, get_history, get_limited_history_text,
#     build_context_text, detect_comparison_intent, resolve_query_with_context
# )
# from logger_config import get_logger

# logger = get_logger(__name__)


# def detect_similar_hotel_intent(query: str) -> bool:
#     """
#     Detect if user wants to find similar hotels.
    
#     Args:
#         query: User query
        
#     Returns:
#         True if similar hotel search is requested
#     """
#     similar_keywords = ["similar", "like this", "alternatives", "other hotels", "comparable", "find hotels like"]
#     query_lower = query.lower()
#     return any(keyword in query_lower for keyword in similar_keywords)




# def format_similar_hotels_response(similar_hotels: List[Document]) -> str:
#     """
#     Format similar hotels for display to user with links and information.
    
#     Args:
#         similar_hotels: List of similar hotel documents
        
#     Returns:
#         Formatted string with hotel information
#     """
#     if not similar_hotels:
#         return "I couldn't find any similar hotels at the moment. Please try a different search."
    
#     response = "Here are some similar hotels you might be interested in:\n\n"
    
#     for i, hotel in enumerate(similar_hotels, 1):
#         # Get hotel information with fallbacks
#         name = hotel.metadata.get("hotel_name") or hotel.metadata.get("official_name") or hotel.metadata.get("name") or "Unknown Hotel"
#         hotel_id = hotel.metadata.get("hotel_id", "N/A")
#         star_rating = hotel.metadata.get("star_rating", "N/A")
#         address = hotel.metadata.get("address", "Address not available")
#         city = hotel.metadata.get("city", "")
#         state = hotel.metadata.get("state", "")
#         overall_rating = hotel.metadata.get("overall_rating", "N/A")
#         total_reviews = hotel.metadata.get("total_reviews", "")
        
#         # Format location
#         location = address
#         if city and state:
#             location += f", {city}, {state}"
#         elif city:
#             location += f", {city}"
        
#         # Format star rating display
#         if star_rating != "N/A" and star_rating:
#             try:
#                 stars = int(float(star_rating))
#                 star_display = f"{'⭐' * stars}"
#             except (ValueError, TypeError):
#                 star_display = str(star_rating)
#         else:
#             star_display = "Rating not available"
        
#         # Truncate description
#         description = hotel.page_content[:200]
#         if len(hotel.page_content) > 200:
#             description += "..."
        
#         # Build hotel entry
#         response += f"**{i}. {name}** ({star_display})\n"
#         response += f"📍 Location: {location}\n"
        
#         if overall_rating != "N/A":
#             response += f"⭐ Guest Rating: {overall_rating}/5"
#             if total_reviews:
#                 response += f" ({total_reviews} reviews)"
#             response += "\n"
        
#         response += f"📝 {description}\n"
#         response += f"🔗 [View this hotel](#hotel/{hotel_id})\n\n"
    
#     return response


# from .langfuse_tracking import track_agent

# @track_agent("comparison_agent")
# async def comparison_node(state: HotelIQState) -> HotelIQState:
#     """
#     Comparison Agent: Handles hotel information retrieval and similar hotel search.
#     """
#     thread_id = state.get("thread_id", "unknown_thread")
#     hotel_id = state.get("hotel_id", "")
#     history_obj = get_history(f"compare_{thread_id}")
#     history_text = get_limited_history_text(history_obj)

#     user_message = state["messages"][-1]["content"]
    
#     logger.info("Comparison Agent processing query", hotel_id=hotel_id)

#     if "metadata" in state and "resolved_query" in state["metadata"]:
#         query_for_retrieval = state["metadata"]["resolved_query"]
#         logger.info("Using resolved query", query=query_for_retrieval)
#     else:
#         query_for_retrieval = user_message
    
#     wants_similar = detect_similar_hotel_intent(query_for_retrieval)

#     if wants_similar:

#         try:
#             similar_hotels = find_similar_hotels(hotel_id, top_k=3, exclude_current=True)
#             answer = format_similar_hotels_response(similar_hotels)
#         except Exception as e:
#             logger.error("Error finding similar hotels", error=str(e))
#             answer = "I apologize, but I encountered an error while searching for similar hotels. Please try again."
#     else:
        
#         try:
#             hotel_info = state.get("metadata", {}).get("hotel_info")
#             hotel_name = state.get("metadata", {}).get("hotel_name", "this hotel")
            
#             if not hotel_info:
#                 answer = f"I don't have information about {hotel_name}. Please try again."
#             else:
#                 context_parts = []
                
#                 context_parts.append(f"Hotel Name: {hotel_info.get('hotel_name', 'N/A')}")
#                 context_parts.append(f"Star Rating: {hotel_info.get('star_rating', 'N/A')}")
#                 context_parts.append(f"Address: {hotel_info.get('address', 'N/A')}, {hotel_info.get('city', '')}, {hotel_info.get('state', '')} {hotel_info.get('zip_code', '')}")
                
#                 if hotel_info.get('phone'):
#                     context_parts.append(f"Phone: {hotel_info.get('phone')}")
#                 if hotel_info.get('website'):
#                     context_parts.append(f"Website: {hotel_info.get('website')}")
#                 if hotel_info.get('total_rooms'):
#                     context_parts.append(f"Total Rooms: {hotel_info.get('total_rooms')}")
#                 if hotel_info.get('overall_rating'):
#                     context_parts.append(f"Overall Rating: {hotel_info.get('overall_rating')}")
                
#                 if hotel_info.get('description'):
#                     context_parts.append(f"\nDescription: {hotel_info.get('description')}")
                
#                 if hotel_info.get('additional_info'):
#                     context_parts.append(f"\nAdditional Information: {hotel_info.get('additional_info')}")
                
#                 # Add amenities information
#                 if hotel_info.get('amenities'):
#                     context_parts.append(f"\nAmenities:{hotel_info.get('amenities')}")
                
#                 # Add policies information
#                 if hotel_info.get('policies'):
#                     context_parts.append(f"\nPolicies:\n{hotel_info.get('policies')}")
                
#                 context_text = "\n".join(context_parts)
                
#                 logger.info("Using CSV data for hotel", hotel_id=hotel_id)
                
#                 answer = comparison_chain.invoke(
#                     {"history": history_text, "context": context_text, "question": user_message}
#                 )
#         except Exception as e:
#             logger.error("Error retrieving hotel information", error=str(e))
#             import traceback
#             traceback.print_exc()
#             answer = "I apologize, but I encountered an error while retrieving hotel information. Please try again."

#     msgs = state.get("messages", [])
#     msgs.append({"role": "assistant", "content": answer})
#     state["messages"] = msgs

#     history_obj.add_user_message(user_message)
#     history_obj.add_ai_message(answer)

#     state["route"] = "end"
#     return state



"""
Comparison Agent
================

Handles hotel information retrieval and similar hotel search queries.
"""

from typing import Dict, List

from langchain_core.documents import Document

from .state import HotelIQState
from .pinecone_retrieval import find_similar_hotels
from .utils import (
    comparison_chain,
    get_history,
    get_limited_history_text,
    build_context_text,
    detect_comparison_intent,
    resolve_query_with_context
)
from logger_config import get_logger

logger = get_logger(__name__)


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

        # Generate full frontend URL for the hotel
        hotel_url = f"https://hotel-iq-765947304209.us-east4.run.app/hotel/{hotel_id}"

        # Build structured hotel entry with each category on its own line
        response += f"**{i}. {name}** {star_display}\n"
        response += f"  • 📍 **Location:** {location}\n"
        
        if overall_rating != "N/A":
            rating_text = f"  • ⭐ **Guest Rating:** {overall_rating}/5"
            if total_reviews:
                rating_text += f" ({total_reviews} reviews)"
            response += f"{rating_text}\n"
        
        response += f"  • 📝 **Description:** {description}\n"
        response += f"  • 🔗 [View this hotel →]({hotel_url})\n\n"

    return response


from .langfuse_tracking import track_agent


@track_agent("comparison_agent")
async def comparison_node(state: HotelIQState) -> HotelIQState:
    """
    Comparison Agent: Handles hotel information retrieval and similar hotel search.
    """
    thread_id = state.get("thread_id", "unknown_thread")
    hotel_id = state.get("hotel_id", "")
    history_obj = get_history(f"compare_{thread_id}")
    history_text = get_limited_history_text(history_obj)
    user_message = state["messages"][-1]["content"]

    logger.info("Comparison Agent processing query", hotel_id=hotel_id)

    if "metadata" in state and "resolved_query" in state["metadata"]:
        query_for_retrieval = state["metadata"]["resolved_query"]
        logger.info("Using resolved query", query=query_for_retrieval)
    else:
        query_for_retrieval = user_message

    wants_similar = detect_similar_hotel_intent(query_for_retrieval)

    if wants_similar:
        try:
            similar_hotels = find_similar_hotels(hotel_id, top_k=3, exclude_current=True)
            answer = format_similar_hotels_response(similar_hotels)
            
            # Populate last_suggestions in state for reference by other agents
            suggestions = []
            for hotel in similar_hotels:
                suggestions.append({
                    "hotel_id": str(hotel.metadata.get("hotel_id", "")),
                    "name": hotel.metadata.get("hotel_name") or hotel.metadata.get("official_name") or hotel.metadata.get("name") or "Unknown Hotel",
                    "star_rating": str(hotel.metadata.get("star_rating", ""))
                })
            state["last_suggestions"] = suggestions
            logger.info("Populated last_suggestions with similar hotels", count=len(suggestions))
            
        except Exception as e:
            logger.error("Error finding similar hotels", error=str(e))
            answer = "I apologize, but I encountered an error while searching for similar hotels. Please try again."
    else:
        try:
            hotel_info = state.get("metadata", {}).get("hotel_info")
            hotel_name = state.get("metadata", {}).get("hotel_name", "this hotel")

            if not hotel_info:
                answer = f"I don't have information about {hotel_name}. Please try again."
            else:
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

                # Add amenities information
                if hotel_info.get('amenities'):
                    context_parts.append(f"\nAmenities:{hotel_info.get('amenities')}")

                # Add policies information
                if hotel_info.get('policies'):
                    context_parts.append(f"\nPolicies:\n{hotel_info.get('policies')}")

                context_text = "\n".join(context_parts)

                logger.info("Using CSV data for hotel", hotel_id=hotel_id)

                answer = comparison_chain.invoke(
                    {"history": history_text, "context": context_text, "question": user_message}
                )

        except Exception as e:
            logger.error("Error retrieving hotel information", error=str(e))
            import traceback
            traceback.print_exc()
            answer = "I apologize, but I encountered an error while retrieving hotel information. Please try again."

    msgs = state.get("messages", [])
    msgs.append({"role": "assistant", "content": answer})
    state["messages"] = msgs

    history_obj.add_user_message(user_message)
    history_obj.add_ai_message(answer)

    state["route"] = "end"

    return state