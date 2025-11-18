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
from .booking_agent import booking_node
from .review_agent import review_node
from .supervisor import supervisor_node


def route_from_metadata(state: HotelIQState) -> str:
    """Metadata agent always routes to supervisor."""
    return "supervisor"


def route_from_supervisor(state: HotelIQState) -> str:
    """Supervisor routes based on detected intent."""
    intent = state.get("intent", "comparison")
    if intent == "booking":
        return "booking"
    elif intent == "review":
        return "review"
    return "comparison"


# Build the agent graph
graph = StateGraph(HotelIQState)

# Add all agent nodes
graph.add_node("metadata_agent", metadata_agent_node)
graph.add_node("supervisor", supervisor_node)
graph.add_node("comparison", comparison_node)
graph.add_node("booking", booking_node)
graph.add_node("review", review_node)

# Set metadata_agent as entry point (it processes references first)
graph.set_entry_point("metadata_agent")

# Define edges
graph.add_edge("metadata_agent", "supervisor")

graph.add_conditional_edges(
    "supervisor",
    route_from_supervisor,
    {
        "comparison": "comparison",
        "booking": "booking",
        "review": "review",
    },
)

graph.add_edge("comparison", END)
graph.add_edge("booking", END)
graph.add_edge("review", END)

# Compile the graph with memory
checkpointer = MemorySaver()
agent_graph = graph.compile(checkpointer=checkpointer)

print("✅ LangGraph graph compiled.")

