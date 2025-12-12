# File: model_pipeline/backend/agents/concierge_agent.py
import os
from typing import Dict, Any, List
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.messages import HumanMessage, AIMessage
from pydantic import BaseModel, Field

# Import from the same package
from .email_service import send_email

# Load API key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# --- SYSTEM PROMPT ---
SYSTEM_PROMPT = """
You are a professional hotel concierge assistant.
Your goal is to assist guests who are currently staying at the hotel.

HOTEL INFORMATION:
{hotel_context}

CLASSIFICATION RULES (Single-word tags):
1. **supplies**: Guest needs items delivered (towels, soap, pillows, water, coffee, extra bed).
2. **cleaning**: Guest needs labor services (clean room, make bed, remove trash).
3. **repair**: Physical hardware is broken (AC, lights, TV, shower, toilet).
4. **wifi**: Technical support only (Slow internet, cannot connect). **DO NOT** use this tag for simple wifi username and password requests.
5. **porter**: Assistance with luggage, bags, valet, car, or taxi/cab booking.
6. **dining**: Guest wants to book a table, order food, room service, or pre-book meals (lunch/dinner).
7. **admin**: Front desk tasks (checkout, extend stay, key cards, billing).
8. **emergency**: DANGER. Fire, medical, 911, police, or safety threats.
9. **other**: General questions, chit-chat, off-topic queries, OR simple information requests (like WiFi password).

PRIORITY RULES:
- 'emergency' is ALWAYS 'high'.
- 'repair' and 'wifi' are 'medium'.
- All others are 'normal'.

GUARDRAILS:
1. Use the HOTEL INFORMATION above to answer questions about amenities, times, and rules.
2. If the info is not in the context, politely say you don't know and offer to ask the front desk.
3. If a user asks about off-topic issues (politics, personal life, coding), politely decline.
4. Keep responses short (max 150 tokens) and professional.
5. **CRITICAL:** If the user asks for the WiFi password and it is listed in HOTEL INFORMATION, provide it immediately. Do NOT tell them to ask the front desk. Do not generate a dummy password.

Current Context:
Guest Name: {guest_name}
Room Number: {room_number}
"""

# --- EMAIL HANDLER ---
def handle_resolution(guest_email, guest_name, request_type, request_details):
    """
    Sends a confirmation email to the guest after their request is logged/resolved.
    """
    # Simple check to avoid sending emails for chit-chat
    if request_type in ["other", "admin", "emergency"]:
        return

    subject = f"HotelIQ: Request Received - {request_type.title()}"
    
    body = f"""
    Dear {guest_name},

    Your request regarding {request_type} has been successfully completed.

    Details: {request_details}

    If you experience any issues or need further assistance, please feel free to let us know, we’re here to help.

    Warm regards,
    HotelIQ Concierge
    """
    
    try:
        # Assuming guest_email is passed correctly; if not available, handle gracefully
        if guest_email:
            send_email(
                to_email=guest_email, 
                subject=subject, 
                body=body
            )
            print(f"Confirmation email sent to {guest_email}")
        else:
            print("No guest email provided; skipping confirmation email.")
    except Exception as e:
        print(f"Failed to send email: {e}")

# --- STRUCTURED OUTPUT ---
class ConciergeResponse(BaseModel):
    response: str = Field(description="The polite response to show to the guest")
    request_type: str = Field(description="One of: 'supplies', 'cleaning', 'repair', 'wifi', 'porter', 'dining', 'admin', 'emergency', 'other'")
    is_service_request: bool = Field(description="True if staff action is needed, False if it's just a question")
    priority: str = Field(description="Priority: 'normal', 'medium', or 'high'", default="normal")

# --- AI CONFIG ---
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.0,
    openai_api_key=OPENAI_API_KEY,
)

parser = JsonOutputParser(pydantic_object=ConciergeResponse)

# We use MessagesPlaceholder to insert the conversation history dynamically
prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    MessagesPlaceholder(variable_name="history"),
    ("user", "{message}\n\n{format_instructions}")
])

chain = prompt | llm | parser

async def process_guest_message(
    message: str, 
    guest_name: str, 
    room_number: str, 
    hotel_context: str, 
    history: List[Dict],
    guest_email: str = None # Added guest_email parameter
) -> Dict[str, Any]:
    """Process guest message with RAG Context and Chat History."""
    
    # Convert simple list of dicts to LangChain Message objects
    chat_history = []
    for msg in history:
        if msg['role'] == 'user':
            chat_history.append(HumanMessage(content=msg['content']))
        else:
            chat_history.append(AIMessage(content=msg['content']))

    try:
        # 1. Invoke the AI Chain
        result = await chain.ainvoke({
            "guest_name": guest_name,
            "room_number": room_number,
            "hotel_context": hotel_context,
            "history": chat_history,
            "message": message,
            "format_instructions": parser.get_format_instructions()
        })
        
        # 2. Trigger Email Notification if it's a valid service request
        if result.get("is_service_request") and result.get("request_type") not in ["other", "emergency"]:
             handle_resolution(
                 guest_email=guest_email, 
                 guest_name=guest_name, 
                 request_type=result.get("request_type"), 
                 request_details=message
             )

        return result
        
    except Exception as e:
        print(f"Concierge Error: {e}")
        return {
            "response": "I apologize, I'm having trouble connecting. Please call the front desk.",
            "request_type": "admin",
            "is_service_request": True,
            "priority": "normal"
        }