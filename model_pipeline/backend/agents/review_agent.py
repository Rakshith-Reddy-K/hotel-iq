"""
Review Agent
============

Handles hotel review queries and provides review summaries.
"""

from typing import List, Dict
from langchain_core.documents import Document

from .state import HotelIQState
from .pinecone_retrieval import retrieve_reviews_by_query, get_reviews_by_hotel_id
from .utils import get_history, get_limited_history_text
from .prompt_loader import get_prompts
from logger_config import get_logger

logger = get_logger(__name__)


def detect_review_summary_intent(query: str) -> bool:
    """
    Detect if user wants a summary of reviews vs specific reviews.
    
    Args:
        query: User query
        
    Returns:
        True if summary is requested, False otherwise
    """
    summary_keywords = ["summary", "summarize", "overview", "general", "overall", "sentiment"]
    query_lower = query.lower()
    return any(keyword in query_lower for keyword in summary_keywords)


def detect_recent_reviews_intent(query: str) -> bool:
    """
    Detect if user wants to see recent/latest reviews.
    
    Args:
        query: User query
        
    Returns:
        True if recent reviews are requested
    """
    recent_keywords = ["recent", "latest", "new", "last", "show me reviews"]
    query_lower = query.lower()
    return any(keyword in query_lower for keyword in recent_keywords)


def format_recent_reviews(reviews: List[Document], limit: int = 3) -> str:
    """
    Format recent reviews for display to user.
    
    Args:
        reviews: List of review documents
        limit: Maximum number of reviews to show
        
    Returns:
        Formatted string with reviews
    """
    if not reviews:
        return "No reviews found for this hotel."
    
    # Take most recent reviews (assuming they are sorted)
    recent_reviews = reviews[:limit]
    
    formatted = "Here are the most recent reviews:\n\n"
    for i, review in enumerate(recent_reviews, 1):
        rating = review.metadata.get("rating", "N/A")
        text = review.page_content[:300]  # Limit review text length
        if len(review.page_content) > 300:
            text += "..."
        
        formatted += f"**Review {i}** (Rating: {rating}/5)\n{text}\n\n"
    
    return formatted


def generate_review_summary(reviews: List[Document], query: str, history_text: str) -> str:
    """
    Generate a summary of reviews using LLM.
    
    Args:
        reviews: List of review documents
        query: User query
        history_text: Conversation history
        
    Returns:
        Generated summary
    """
    if not reviews:
        return "No reviews found for this hotel."
    
    review_texts = []
    for review in reviews[:20]:  # Limit to 20 reviews for summary
        rating = review.metadata.get("rating", "N/A")
        review_texts.append(f"Rating: {rating}/5\n{review.page_content}")
    
    context = "\n\n---\n\n".join(review_texts)
    
    from .utils import comparison_chain
    
    summary_prompt = f"Based on the following hotel reviews, {query}\n\nReviews:\n{context}"
    
    try:
        summary = comparison_chain.invoke({
            "history": history_text,
            "context": context,
            "question": query
        })
        return summary
    except Exception as e:
        logger.error("Error generating review summary", error=str(e))
        return "I apologize, but I encountered an error while summarizing the reviews. Please try again."


from .langfuse_tracking import track_agent

@track_agent("review_agent")
async def review_node(state: HotelIQState) -> HotelIQState:
    """
    Review Agent: Handles review-specific queries for a hotel.
    
    Features:
    - Filters reviews by hotel_id
    - Shows 2-3 recent reviews on request
    - Generates review summaries on request
    - Answers specific questions about reviews
    """
    thread_id = state.get("thread_id", "unknown_thread")
    hotel_id = state.get("hotel_id", "")
    user_message = state["messages"][-1]["content"]
    
    history_obj = get_history(f"compare_{thread_id}")
    history_text = get_limited_history_text(history_obj)
    
    logger.info("Review Agent processing query", hotel_id=hotel_id)
    
    try:
        if detect_recent_reviews_intent(user_message):
            hotel_reviews = get_reviews_by_hotel_id(hotel_id, top_k=20)
            logger.info("Found reviews for hotel", count=len(hotel_reviews), hotel_id=hotel_id)
            
            answer = format_recent_reviews(hotel_reviews, limit=3)
            
        elif detect_review_summary_intent(user_message):
            hotel_reviews = get_reviews_by_hotel_id(hotel_id, top_k=50)
            logger.info("Found reviews for hotel", count=len(hotel_reviews), hotel_id=hotel_id)
            
            answer = generate_review_summary(hotel_reviews, user_message, history_text)
        else:
            hotel_reviews = retrieve_reviews_by_query(
                query=user_message,
                hotel_id=hotel_id,
                top_k=20
            )
            logger.info("Found relevant reviews for hotel", count=len(hotel_reviews), hotel_id=hotel_id)
            
            answer = generate_review_summary(hotel_reviews, user_message, history_text)
        
    except Exception as e:
        logger.error("Error retrieving reviews", error=str(e))
        answer = "I apologize, but I encountered an error while retrieving reviews. Please try again."
    
    msgs = state.get("messages", [])
    msgs.append({"role": "assistant", "content": answer})
    state["messages"] = msgs
    
    history_obj.add_user_message(user_message)
    history_obj.add_ai_message(answer)
    
    state["route"] = "end"
    return state

