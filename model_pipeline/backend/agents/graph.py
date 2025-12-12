

"""
Agent Graph Setup
=================

Configures and compiles the LangGraph agent workflow.
"""

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from .state import HotelIQState
from .query_agent import metadata_agent_node
from .comparison_agent import comparison_node
from .booking_collection_agent import booking_collection_node
from .booking_execution_agent import booking_execution_node
from .review_agent import review_node
from .supervisor import supervisor_node
from logger_config import get_logger

logger = get_logger(__name__)


def route_from_metadata(state: HotelIQState) -> str:
    """Metadata agent always routes to supervisor."""
    return "supervisor"


def route_from_supervisor(state: HotelIQState) -> str:
    """
    Supervisor routes based on detected intent.

    Booking-aware routing:
    - If there is an active booking_conversation in a mid-booking stage
      (choosing_room_type / collecting / confirming), we ALWAYS continue
      to booking_collection, even if the LLM says the intent is COMPARISON.
    """
    # 1) If a booking flow is already in progress, keep the user in it
    booking_conv = state.get("booking_conversation")
    if isinstance(booking_conv, dict):
        stage = (booking_conv.get("stage") or "").lower()

        if stage in {"choosing_room_type", "collecting", "confirming"}:
            logger.info(
                "Continuing booking flow from supervisor",
                stage=stage,
            )
            return "booking_collection"

    # 2) Otherwise, fall back to intent-based routing
    intent = (state.get("intent") or "comparison").lower()

    if intent == "booking":
        return "booking_collection"
    elif intent == "review":
        return "review"
    else:
        return "comparison"


def route_from_booking(state: HotelIQState) -> str:
    """
    Route from booking collection based on stage.

    booking_collection_agent should set:
        state["booking_conversation"]["stage"] to one of:
        - 'initial' / 'choosing_room_type' / 'collecting' / 'confirming' → stay in collection (END)
        - 'executing'                                                    → go to booking_execution
    """
    booking_state = state.get("booking_conversation") or {}
    stage = (booking_state.get("stage") or "collecting").lower()

    if stage == "executing":
        return "booking_execution"
    else:
        # Finish this turn; next user message will re-enter via supervisor
        return "end"


# ---------------------------------------------------------------------
# Graph definition
# ---------------------------------------------------------------------

graph = StateGraph(HotelIQState)

# Add all nodes
graph.add_node("metadata_agent", metadata_agent_node)
graph.add_node("supervisor", supervisor_node)
graph.add_node("comparison", comparison_node)
graph.add_node("booking_collection", booking_collection_node)
graph.add_node("booking_execution", booking_execution_node)
graph.add_node("review", review_node)

# Set entry point
graph.set_entry_point("metadata_agent")

# Flow: metadata -> supervisor
graph.add_edge("metadata_agent", "supervisor")

# Supervisor routing (booking-aware)
graph.add_conditional_edges(
    "supervisor",
    route_from_supervisor,
    {
        "comparison": "comparison",
        "booking_collection": "booking_collection",
        "review": "review",
    },
)

# Booking collection routing
graph.add_conditional_edges(
    "booking_collection",
    route_from_booking,
    {
        "booking_execution": "booking_execution",
        "end": END,
    },
)

# End nodes
graph.add_edge("comparison", END)
graph.add_edge("booking_execution", END)
graph.add_edge("review", END)

checkpointer = MemorySaver()
agent_graph = graph.compile(checkpointer=checkpointer)

logger.info("LangGraph graph compiled with booking agents.")
