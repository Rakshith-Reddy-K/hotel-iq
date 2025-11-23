"""
Review Agent
============

Handles hotel review queries and provides review summaries.
"""

from typing import List, Dict
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate  
from langchain_core.output_parsers import StrOutputParser

from .state import HotelIQState
from .config import llm
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
    Generate a summary of reviews using LLM with rating awareness to reduce bias.
    """
    if not reviews:
        return "No reviews found for this hotel."
    
    # --- NEW LOGIC: Calculate stats for the prompt ---
    ratings = []
    review_texts = []
    for review in reviews[:20]:  # Limit to 20 reviews
        r_val = review.metadata.get("rating") or review.metadata.get("overall_rating")
        
        # Try to parse rating
        if r_val and str(r_val) != "N/A":
            try:
                ratings.append(float(r_val))
            except ValueError:
                pass
        
        review_texts.append(f"Rating: {r_val}/5\n{review.page_content}")
    
    avg_rating = f"{sum(ratings)/len(ratings):.1f}" if ratings else "N/A"
    context = "\n\n---\n\n".join(review_texts)

    # --- NEW LOGIC: Use specific prompt chain instead of comparison_chain ---
    prompts = get_prompts()
    prompt_text = prompts.get("review_agent.summary_prompt")
    
    # Fallback if YAML isn't updated yet
    if not prompt_text:
        prompt_text = "Summarize these reviews ({avg_rating}/5 avg): {context}\nQuery: {query}"

    summary_chain = ChatPromptTemplate.from_template(prompt_text) | llm | StrOutputParser()
    
    try:
        summary = summary_chain.invoke({
            "hotel_name": "this hotel", 
            "avg_rating": avg_rating,
            "context": context,
            "query": query
        })
        return summary
    except Exception as e:
        logger.error("Error generating review summary", error=str(e))
        return "I apologize, but I encountered an error while summarizing the reviews."

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
            context_text = "\n".join([d.page_content for d in hotel_reviews[:3]])
        elif detect_review_summary_intent(user_message):
            hotel_reviews = get_reviews_by_hotel_id(hotel_id, top_k=50)
            logger.info("Found reviews for hotel", count=len(hotel_reviews), hotel_id=hotel_id)
            
            answer = generate_review_summary(hotel_reviews, user_message, history_text)
            context_text = "\n".join([d.page_content for d in hotel_reviews[:20]])
        else:
            hotel_reviews = retrieve_reviews_by_query(
                query=user_message,
                hotel_id=hotel_id,
                top_k=20
            )
            logger.info("Found relevant reviews for hotel", count=len(hotel_reviews), hotel_id=hotel_id)
            
            answer = generate_review_summary(hotel_reviews, user_message, history_text)
            context_text = "\n".join([d.page_content for d in hotel_reviews[:20]])
    except Exception as e:
        logger.error("Error retrieving reviews", error=str(e))
        answer = "I apologize, but I encountered an error while retrieving reviews. Please try again."
    
    msgs = state.get("messages", [])
    msgs.append({"role": "assistant", "content": answer})
    state["messages"] = msgs
    
    state["retrieved_context"] = context_text
    
    history_obj.add_user_message(user_message)
    history_obj.add_ai_message(answer)
    
    state["route"] = "end"
    return state

