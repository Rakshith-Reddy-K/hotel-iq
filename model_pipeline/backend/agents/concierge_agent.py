import os
from typing import Dict, Any
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field

# Load API key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# --- SYSTEM PROMPT (From Doc) ---
SYSTEM_PROMPT = """
You are a professional hotel concierge assistant.
Your goal is to assist guests who are currently staying at the hotel.

GUARDRAILS:
1. You answer ONLY hotel service questions (Housekeeping, Amenities, Dining, Checkout).
2. If a user asks about off-topic issues (politics, personal life, coding, jokes), politely decline.
3. Keep responses short (max 150 tokens) and professional.
4. Determine if the user has a specific request (like needing towels) or just a question.

Current Context:
Guest Name: {guest_name}
Room Number: {room_number}
"""

# --- STRUCTURED OUTPUT ---
class ConciergeResponse(BaseModel):
    response: str = Field(description="The polite response to show to the guest")
    request_type: str = Field(description="Type: 'housekeeping', 'amenity', 'info', 'checkout', 'complaint', or 'other'")
    is_service_request: bool = Field(description="True if this requires staff action (e.g. asking for towels), False if just a question")
    priority: str = Field(description="Priority: 'normal' or 'high'", default="normal")

# --- AI CONFIG ---
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.0,
    openai_api_key=OPENAI_API_KEY,
    # callbacks=[langfuse_handler],
)

parser = JsonOutputParser(pydantic_object=ConciergeResponse)

prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("user", "{message}\n\n{format_instructions}")
])

chain = prompt | llm | parser

async def process_guest_message(message: str, guest_name: str, room_number: str) -> Dict[str, Any]:
    """Process guest message and return structured data for DB."""
    try:
        result = await chain.ainvoke({
            "guest_name": guest_name,
            "room_number": room_number,
            "message": message,
            "format_instructions": parser.get_format_instructions()
        })
        return result
    except Exception as e:
        print(f"Concierge Error: {e}")
        return {
            "response": "I apologize, I'm having trouble connecting. Please call the front desk.",
            "request_type": "error",
            "is_service_request": False,
            "priority": "normal"
        }