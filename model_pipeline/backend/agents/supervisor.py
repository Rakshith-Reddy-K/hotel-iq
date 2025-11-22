"""
Supervisor Agent
================

Routes requests to appropriate agents based on LLM-powered intent detection.
"""

from typing import Literal

from .state import HotelIQState
from .config import llm
from .prompt_loader import get_prompts
from logger_config import get_logger

logger = get_logger(__name__)


async def detect_intent_with_llm(query: str) -> str:
    """
    Use LLM to detect user intent and determine which agent should handle the query.
    
    Args:
        query: User query
        
    Returns:
        Intent string: "review", "booking", or "comparison"
    """
    prompts = get_prompts()
    
    prompt = prompts.format("supervisor.intent_classification", query=query)
    
    response = await llm.ainvoke(prompt)
    intent_raw = response.content.strip().upper()
    
    logger.info("LLM Intent Classification", intent=intent_raw)
    
    if "REVIEW" in intent_raw:
        return "review"
    elif "BOOKING" in intent_raw:
        return "booking"
    elif "COMPARISON" in intent_raw:
        return "comparison"
    else:
        logger.warning("Unexpected LLM response, defaulting to comparison", response=intent_raw)
        return "comparison"


from .langfuse_tracking import track_agent

@track_agent("supervisor_agent")
async def supervisor_node(state: HotelIQState) -> HotelIQState:
    """
    Supervisor Agent: Routes user queries to the appropriate specialized agent using LLM.
    
    Decision Logic (via LLM):
    - Review intent → routes to review_agent
    - Booking intent → routes to booking_agent
    - Comparison intent → routes to comparison_agent (hotel info, similar hotels)
    
    Uses the resolved query from metadata agent when available.
    """
    if "metadata" in state and "resolved_query" in state["metadata"]:
        last_msg = state["metadata"]["resolved_query"]
    else:
        last_msg = state["messages"][-1]["content"]
    
    logger.info("Supervisor analyzing query", query=last_msg[:100])
    
    intent_str = await detect_intent_with_llm(last_msg)
    
    if intent_str == "review":
        intent: Literal["review"] = "review"
        logger.info("Routing to REVIEW agent")
    elif intent_str == "booking":
        intent: Literal["booking"] = "booking"
        logger.info("Routing to BOOKING agent")
    else:
        intent: Literal["comparison"] = "comparison"
        logger.info("Routing to COMPARISON agent")
    
    state["intent"] = intent
    state["route"] = intent
    return state

