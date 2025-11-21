"""
Langfuse Tracker
================

Utilities for tracking LLM calls and agent execution with Langfuse.
"""
from langfuse import Langfuse
from langfuse.decorators import observe, langfuse_context
import os
from typing import Any, Dict, Optional
from functools import wraps

# Initialize Langfuse client
langfuse = Langfuse(
    secret_key="sk-lf-3ce79b22-618a-4df4-9fe5-f4f70a233142",
    public_key="pk-lf-9077d3b2-3330-4b41-8862-f8bd96abdf9d",
    host="http://136.115.66.228:3000"
)

def track_agent(agent_name: str):
    """
    Decorator to track agent execution with query, response, and token usage
    
    Args:
        agent_name: Name of the agent (e.g., 'query_agent', 'review_agent')
    """
    def decorator(func):
        @observe(name=agent_name)
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Extract query from args/kwargs
            query = kwargs.get('query') or (args[0] if args else None)
            
            # Update observation with input
            langfuse_context.update_current_observation(
                input={"query": str(query), "agent": agent_name},
                metadata={"agent_type": agent_name}
            )
            
            # Execute the agent function
            result = func(*args, **kwargs)
            
            # Update observation with output
            langfuse_context.update_current_observation(
                output={"response": str(result)}
            )
            
            return result
        return wrapper
    return decorator

def track_llm_call(
    prompt: str,
    model: str,
    response: str,
    usage: Optional[Dict[str, int]] = None,
    metadata: Optional[Dict[str, Any]] = None
):
    """
    Track LLM calls with token usage
    
    Args:
        prompt: The prompt sent to LLM
        model: Model identifier
        response: LLM response
        usage: Token usage dict with 'prompt_tokens', 'completion_tokens', 'total_tokens'
        metadata: Additional metadata
    """
    langfuse_context.update_current_observation(
        model=model,
        input=prompt,
        output=response,
        usage=usage or {},
        metadata=metadata or {}
    )
    
    return response

def flush_langfuse():
    """Flush all pending Langfuse events"""
    langfuse.flush()