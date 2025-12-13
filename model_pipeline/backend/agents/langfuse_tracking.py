from langfuse import observe, Langfuse
import os
from typing import Any, Dict, Optional
from functools import wraps
import inspect

# Initialize Langfuse client
langfuse = Langfuse(
    secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
    public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
    host=os.getenv("LANGFUSE_HOST")
)

def track_agent(agent_name: str):
    """
    Decorator to track agent execution with query, response, and token usage
    
    Args:
        agent_name: Name of the agent (e.g., 'query_agent', 'review_agent')
    """
    def decorator(func):
        if inspect.iscoroutinefunction(func):
            @observe(name=agent_name)
            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                # Execute the agent function
                result = await func(*args, **kwargs)
                return result
            return async_wrapper
        else:
            @observe(name=agent_name)
            @wraps(func)
            def sync_wrapper(*args, **kwargs):
                # Execute the agent function
                result = func(*args, **kwargs)
                return result
            return sync_wrapper
    return decorator

def flush_langfuse():
    """Flush all pending Langfuse events"""
    langfuse.flush()