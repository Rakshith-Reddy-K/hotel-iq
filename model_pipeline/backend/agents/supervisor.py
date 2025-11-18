"""
Supervisor Agent
================

Routes requests to appropriate agents based on LLM-powered intent detection.
"""

from typing import Literal

from .state import HotelIQState
from .config import llm
from .prompt_loader import get_prompts


def detect_intent_with_llm(query: str) -> str:
    """
    Use LLM to detect user intent and determine which agent should handle the query.
    
    Args:
        query: User query
        
    Returns:
        Intent string: "review", "booking", or "comparison"
    """
    prompts = get_prompts()
    
    # Get the intent classification prompt
    prompt = prompts.format("supervisor.intent_classification", query=query)
    
    # Call LLM to classify intent
    response = llm.invoke(prompt)
    intent_raw = response.content.strip().upper()
    
    print(f"🤖 LLM Intent Classification: '{intent_raw}'")
    
    # Map LLM response to valid intent
    if "REVIEW" in intent_raw:
        return "review"
    elif "BOOKING" in intent_raw:
        return "booking"
    elif "COMPARISON" in intent_raw:
        return "comparison"
    else:
        # Default to comparison if unclear
        print(f"⚠️ Unexpected LLM response: '{intent_raw}', defaulting to comparison")
        return "comparison"


def supervisor_node(state: HotelIQState) -> HotelIQState:
    """
    Supervisor Agent: Routes user queries to the appropriate specialized agent using LLM.
    
    Decision Logic (via LLM):
    - Review intent → routes to review_agent
    - Booking intent → routes to booking_agent
    - Comparison intent → routes to comparison_agent (hotel info, similar hotels)
    
    Uses the resolved query from metadata agent when available.
    """
    # Use resolved query if available from metadata agent
    if "metadata" in state and "resolved_query" in state["metadata"]:
        last_msg = state["metadata"]["resolved_query"]
    else:
        last_msg = state["messages"][-1]["content"]
    
    print(f"🧭 Supervisor analyzing query: '{last_msg[:100]}...'")
    
    # Use LLM to detect intent
    intent_str = detect_intent_with_llm(last_msg)
    
    # Convert to Literal type for type checking
    if intent_str == "review":
        intent: Literal["review"] = "review"
        print("→ Routing to REVIEW agent")
    elif intent_str == "booking":
        intent: Literal["booking"] = "booking"
        print("→ Routing to BOOKING agent")
    else:
        intent: Literal["comparison"] = "comparison"
        print("→ Routing to COMPARISON agent")
    
    state["intent"] = intent
    state["route"] = intent
    return state

