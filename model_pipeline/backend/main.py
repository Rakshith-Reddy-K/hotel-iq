"""
HotelIQ Backend Server
======================

FastAPI application for the HotelIQ hotel comparison and booking chatbot.
"""

import os
from dotenv import load_dotenv 

# Load .env from the backend folder
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(BASE_DIR, ".env"))

import uuid
import logging
from pathlib import Path
from typing import List, Optional
from dataclasses import dataclass

from fastapi import FastAPI, HTTPException, Depends, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel, Field
from contextlib import asynccontextmanager

from agents import agent_graph
from agents.state import HotelIQState
from agents.validation import sanitize_user_input
import bucket_util
import path as path_util

from agents.concierge_agent import process_guest_message
from agents.pinecone_retrieval import get_hotel_by_id  
from logger_config import configure_logger, get_logger
import json
from datetime import datetime
from fastapi import HTTPException, status
from typing import Dict, Any
from sql.queries import list_tables
from auth_routes import auth_router
from hotel_routes import hotel_router
from sql.db_pool import get_connection  # ADD THIS IMPORT

# Setup logging
configure_logger()
logger = get_logger(__name__)

# ======================================================
# GCP CREDENTIALS SETUP
# ======================================================

# Set GCP credentials path if not already set
if not os.getenv('GOOGLE_APPLICATION_CREDENTIALS'):
    credentials_path = Path(__file__).parent / "config" / "gcp-service-account.json"
    if credentials_path.exists():
        os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = str(credentials_path)
        logger.info("GCP credentials loaded", path=str(credentials_path))
    else:
        logger.warning("GCP credentials file not found", path=str(credentials_path))

# ======================================================
# CONFIGURATION
# ======================================================

current_dir = Path(__file__).resolve().parent
if (current_dir / "frontend").exists():
    FRONTEND_DIR = current_dir / "frontend"
elif (current_dir.parent / "frontend").exists():
    FRONTEND_DIR = current_dir.parent / "frontend"
else:
    logger.warning("Frontend directory not found!")
    FRONTEND_DIR = current_dir / "frontend"

# ======================================================
# DATA INITIALIZATION
# ======================================================

def check_data_files_exist(city: str) -> dict:
    """Check if all required data files exist locally."""
    files_status = {
        'hotels': Path(path_util.get_processed_hotels_path(city)).exists(),
        'amenities': Path(path_util.get_processed_amenities_path(city)).exists(),
        'policies': Path(path_util.get_processed_policies_path(city)).exists(),
        'reviews': Path(path_util.get_processed_reviews_path(city)).exists(),
    }
    return files_status


def download_processed_data():
    """Download processed CSV files from GCS to local data folder on startup."""
    city = os.getenv('CITY', 'boston')
    logger.info("Checking data files", city=city)
    
    files_status = check_data_files_exist(city)
    all_exist = all(files_status.values())
    
    if all_exist:
        logger.info("All data files already exist locally. Skipping download.", location=str(Path(path_util.get_processed_dir(city))))
        for file_name in files_status.keys():
            logger.info(f"   ✓ {file_name}.csv")
        return True
    
    missing_files = [name for name, exists in files_status.items() if not exists]
    logger.info("Downloading missing data files", missing_files=missing_files)
    
    files_to_download = [
        ('hotels', path_util.get_gcs_processed_table_path(city, 'hotels'), 
         path_util.get_processed_hotels_path(city)),
        ('amenities', path_util.get_gcs_processed_table_path(city, 'amenities'), 
         path_util.get_processed_amenities_path(city)),
        ('policies', path_util.get_gcs_processed_table_path(city, 'policies'), 
         path_util.get_processed_policies_path(city)),
        ('reviews', path_util.get_gcs_processed_table_path(city, 'reviews'), 
         path_util.get_processed_reviews_path(city)),
    ]
    
    success_count = 0
    skipped_count = 0
    for file_name, gcs_path, local_path in files_to_download:
        try:
            if Path(local_path).exists():
                logger.info(f"Skipping {file_name}.csv (already exists)")
                skipped_count += 1
                success_count += 1
                continue
            
            logger.info(f"Downloading {file_name}.csv...")
            bucket_util.download_file_from_gcs(gcs_path, local_path)
            logger.info(f"Successfully downloaded {file_name}.csv")
            success_count += 1
        except Exception as e:
            logger.error(f"Failed to download {file_name}.csv", error=str(e))
    
    if skipped_count > 0:
        logger.info("Download summary", downloaded=success_count - skipped_count, total=len(files_to_download), skipped=skipped_count)
    else:
        logger.info("Download summary", downloaded=success_count, total=len(files_to_download))
    
    return success_count == len(files_to_download)

# ======================================================
# PYDANTIC MODELS
# ======================================================

class MessageModel(BaseModel):
    """Model for a chat message."""
    id: int
    from_: str = Field(..., alias="from")
    text: str
    timestamp: str

    class Config:
        populate_by_name = True


class SaveConversationRequest(BaseModel):
    """Request model for saving conversation."""
    threadId: str = Field(..., max_length=200)
    userId: str = Field(..., max_length=100)
    hotelId: str = Field(..., max_length=50)
    messages: List[MessageModel]


class ConversationResponse(BaseModel):
    """Response model for conversation retrieval."""
    threadId: str
    userId: str
    hotelId: str
    messages: List[Dict[str, Any]]
    lastUpdated: str


class DeleteConversationRequest(BaseModel):
    """Request model for deleting conversation."""
    userId: str = Field(..., max_length=100)
    hotelId: str = Field(..., max_length=50)


class GuestVerifyRequest(BaseModel):
    hotelId: int 
    roomNumber: str
    lastName: str


class GuestInfoRequest(BaseModel):
    """Request model for getting guest info by room number."""
    hotelId: int
    roomNumber: str


class BookingLoginRequest(BaseModel):
    """Request model for guest login with booking reference."""
    bookingReference: str


class GuestChatRequest(BaseModel):
    bookingId: int
    hotelId: str 
    roomNumber: str
    guestName: str
    message: str
    bookingReference: Optional[str] = None  # ADD THIS


class RequestUpdate(BaseModel):
    status: str
    assigned_to: Optional[str] = None


class BookingStatusUpdate(BaseModel):
    """Request model for updating booking check-in status."""
    status: str


class ChatRequest(BaseModel):
    """Request model for chat endpoint with validation."""
    message: str = Field(..., min_length=1, max_length=2000)
    user_id: str = Field(..., min_length=1, max_length=100)
    hotel_id: str = Field(..., min_length=1, max_length=50)
    thread_id: Optional[str] = Field(None, max_length=200)


class ChatResponseModel(BaseModel):
    """Response model for chat endpoint."""
    response: str
    thread_id: str
    followup_suggestions: List[str]
# Route to return chat.html UI
# @app.get("/chat")
# async def chat_page():
#     """Serve the chat interface HTML page."""
#     return FileResponse(FRONTEND_DIR / "chat.html")

from fastapi.responses import HTMLResponse

@app.get("/chat", response_class=HTMLResponse)
async def chat_page():
    """Serve the chat interface HTML page."""
    html_path = FRONTEND_DIR / "chat.html"

    if not html_path.exists():
        logger.error(f"chat.html not found at {html_path}")
        return HTMLResponse(
            content=f"<h1>chat.html not found</h1><p>Looked in: {html_path}</p>",
            status_code=500,
        )

    with open(html_path, "r", encoding="utf-8") as f:
        return f.read()

# ======================================================
# CHAT SERVICE
# ======================================================

@dataclass
class ChatServiceResponse:
    """Response from the chat service."""
    response: str
    thread_id: str
    followup_suggestions: List[str]


class ChatService:
    """Service for processing chat messages through the agent graph."""
    
    def __init__(self, agent_graph):
        self.agent_graph = agent_graph

    async def process_message(
        self,
        message: str,
        thread_id: Optional[str],
        user_id: str,
        hotel_id: str,
    ) -> ChatServiceResponse:
        """Process a user message through the agent graph."""
        if not thread_id:
            thread_id = f"thread_{user_id}_{uuid.uuid4()}"

        messages = []
        try:
            config = {"configurable": {"thread_id": thread_id}}
            previous_state = self.agent_graph.get_state(config)
            
            if previous_state and previous_state.values:
                previous_messages = previous_state.values.get("messages", [])
                if previous_messages:
                    messages = previous_messages.copy()
                    logger.info("Retrieved previous messages", count=len(messages), thread_id=thread_id)
        except Exception as e:
            logger.warning("Could not retrieve previous state", error=str(e))
        
        messages.append({"role": "user", "content": message})

        init_state: HotelIQState = {
            "messages": messages,
            "user_id": user_id,
            "thread_id": thread_id,
            "hotel_id": hotel_id,
        }

        result_state: HotelIQState = await self.agent_graph.ainvoke(
            init_state,
            config={"configurable": {"thread_id": thread_id}},
        )

        assistant_msg = result_state["messages"][-1]["content"]

        followups = [
            "Show me similar hotels with a different budget.",
            "Tell me more about the area around the recommended hotel.",
            "What are some family-friendly attractions nearby?",
        ]

        return ChatServiceResponse(
            response=assistant_msg,
            thread_id=thread_id,
            followup_suggestions=followups,
        )


chat_service = ChatService(agent_graph)
logger.info("ChatService ready.")


# ======================================================
# FASTAPI APP SETUP
# ======================================================

app = FastAPI(title="HotelIQ Comparison API")

@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "HotelIQ API is running",
        "status": "ok",
        "service": "hoteliq-backend",
        "version": "1.0.0"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint for Docker and Cloud Run."""
    return {
        "status": "healthy",
        "service": "hoteliq-backend",
        "version": "1.0.0"
    }

@app.get("/health/ready")
async def readiness_check():
    """Readiness check - verifies app is ready to serve traffic."""
    city = os.getenv('CITY', 'boston')
    files_status = check_data_files_exist(city)
    all_ready = all(files_status.values())
    
    return {
        "status": "ready" if all_ready else "not_ready",
        "service": "hoteliq-backend",
        "data_files": files_status,
        "city": city
    }

@app.on_event("startup")
async def startup_event():
    """Execute on application startup."""
    logger.info("Starting HotelIQ API...")
    download_processed_data()
    
    # Initialize database connection pool
    try:
        from sql.db_pool import initialize_pool
        initialize_pool()
        logger.info("✅ Database connection pool initialized")
    except Exception as e:
        logger.error(f"❌ DB Connection Failed: {e}")
    
    logger.info("Startup complete!")

@app.on_event("shutdown")
async def shutdown_event():
    """Close database connections on shutdown."""
    try:
        from sql.db_pool import close_all
        close_all()
        logger.info("🛑 Database connections closed")
    except Exception as e:
        logger.error(f"Error closing connections: {e}")

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth_router, prefix="/api/v1")
app.include_router(hotel_router, prefix="/api/v1")

if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")
else:
    logger.warning("Frontend directory not found", path=str(FRONTEND_DIR))


@app.get("/chat", response_class=HTMLResponse)
async def chat_page():
    """Serve the chat interface HTML page."""
    html_path = FRONTEND_DIR / "chat.html"
    if not html_path.exists():
        return HTMLResponse(
            content=f"<h1>chat.html not found</h1><p>Looked in: {html_path}</p>",
            status_code=500,
        )
    with open(html_path, "r", encoding="utf-8") as f:
        return f.read()


@app.get("/admin")
async def admin_page():
    """Serve the admin dashboard HTML page."""
    return FileResponse(FRONTEND_DIR / "admin.html")


# ======================================================
# CHAT API ROUTES
# ======================================================

@app.post("/api/v1/chat/message", response_model=ChatResponseModel)
async def send_message(request: ChatRequest):
    """Process a chat message."""
    sanitized_message = sanitize_user_input(request.message)
    
    res = await chat_service.process_message(
        message=sanitized_message,
        thread_id=request.thread_id,
        user_id=request.user_id,
        hotel_id=request.hotel_id,
    )
    return ChatResponseModel(
        response=res.response,
        thread_id=res.thread_id,
        followup_suggestions=res.followup_suggestions,
    )


@app.post("/api/v1/chat/save")
async def save_conversation(request: SaveConversationRequest):
    """Save chat conversation to Google Cloud Storage."""
    try:
        thread_id = sanitize_user_input(request.threadId)
        user_id = sanitize_user_input(request.userId)
        hotel_id = sanitize_user_input(request.hotelId)
        
        conversation_data = {
            "threadId": thread_id,
            "userId": user_id,
            "hotelId": hotel_id,
            "messages": [msg.dict(by_alias=True) for msg in request.messages],
            "lastUpdated": datetime.utcnow().isoformat(),
        }
        
        blob_name = f"conversations/{hotel_id}/{user_id}/{thread_id}.json"
        temp_dir = Path("./temp")
        temp_dir.mkdir(exist_ok=True)
        temp_file = temp_dir / f"{thread_id}.json"
        
        with open(temp_file, 'w') as f:
            json.dump(conversation_data, f, indent=2)
        
        bucket_util.upload_file_to_gcs(str(temp_file), blob_name)
        temp_file.unlink()
        
        logger.info("Conversation saved", thread_id=thread_id, user_id=user_id)
        
        return {
            "status": "success",
            "message": "Conversation saved successfully",
            "threadId": thread_id,
            "blobPath": blob_name
        }
    except Exception as e:
        logger.error("Error saving conversation", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to save conversation: {str(e)}"
        )


@app.get("/api/v1/chat/conversation/{thread_id}", response_model=ConversationResponse)
async def get_conversation(thread_id: str, userId: str, hotelId: str):
    """Retrieve a chat conversation from Google Cloud Storage."""
    try:
        thread_id = sanitize_user_input(thread_id)
        user_id = sanitize_user_input(userId)
        hotel_id = sanitize_user_input(hotelId)
        
        blob_name = f"conversations/{hotel_id}/{user_id}/{thread_id}.json"
        temp_dir = Path("./temp")
        temp_dir.mkdir(exist_ok=True)
        temp_file = temp_dir / f"{thread_id}_download.json"
        
        try:
            bucket_util.download_file_from_gcs(blob_name, str(temp_file))
        except FileNotFoundError:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Conversation not found")
        
        with open(temp_file, 'r') as f:
            conversation_data = json.load(f)
        
        temp_file.unlink()
        logger.info("Conversation retrieved", thread_id=thread_id, user_id=user_id)
        
        return ConversationResponse(**conversation_data)
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error retrieving conversation", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve conversation: {str(e)}"
        )


@app.delete("/api/v1/chat/conversation/{thread_id}")
async def delete_conversation(thread_id: str, request: DeleteConversationRequest):
    """Delete a chat conversation from Google Cloud Storage."""
    try:
        thread_id = sanitize_user_input(thread_id)
        user_id = sanitize_user_input(request.userId)
        hotel_id = sanitize_user_input(request.hotelId)
        
        blob_name = f"conversations/{hotel_id}/{user_id}/{thread_id}.json"
        bucket = bucket_util.get_bucket()
        blob = bucket.blob(blob_name)
        
        if not blob.exists():
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Conversation not found")
        
        blob.delete()
        logger.info("Conversation deleted", thread_id=thread_id, user_id=user_id)
        
        return {
            "status": "success",
            "message": "Conversation deleted successfully",
            "threadId": thread_id
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error deleting conversation", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete conversation: {str(e)}"
        )


@app.get("/api/v1/chat/conversations/list")
async def list_conversations(userId: str, hotelId: str, limit: int = 50):
    """List all conversations for a user at a specific hotel."""
    try:
        user_id = sanitize_user_input(userId)
        hotel_id = sanitize_user_input(hotelId)
        
        prefix = f"conversations/{hotel_id}/{user_id}/"
        bucket = bucket_util.get_bucket()
        blobs = list(bucket.list_blobs(prefix=prefix, max_results=limit))
        
        conversations = []
        for blob in blobs:
            thread_id = blob.name.split('/')[-1].replace('.json', '')
            conversations.append({
                "threadId": thread_id,
                "lastModified": blob.updated.isoformat() if blob.updated else None,
                "size": blob.size
            })
        
        logger.info("Listed conversations", user_id=user_id, count=len(conversations))
        
        return {
            "status": "success",
            "conversations": conversations,
            "count": len(conversations)
        }
    except Exception as e:
        logger.error("Error listing conversations", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list conversations: {str(e)}"
        )


# ======================================================
# CONCIERGE API ROUTES - GUEST ACCESS
# ======================================================

@app.post("/api/guest/booking-login")
async def guest_booking_login(req: BookingLoginRequest):
    """Guest login using booking reference number."""
    try:
        with get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute("""
                    SELECT id, hotel_id, room_number, guest_first_name, guest_last_name, 
                           check_in_date, check_out_date, check_in_status, hotel_name,
                           guest_email, num_guests, room_type, booking_reference
                    FROM bookings 
                    WHERE LOWER(booking_reference) = LOWER(%s)
                    ORDER BY check_in_date DESC
                    LIMIT 1
                """, (req.bookingReference.strip(),))
                
                row = cursor.fetchone()
                
                if row:
                    booking_data = {
                        'id': row[0], 'hotel_id': row[1], 'room_number': row[2],
                        'guest_first_name': row[3], 'guest_last_name': row[4],
                        'check_in_date': row[5], 'check_out_date': row[6],
                        'check_in_status': row[7], 'hotel_name': row[8],
                        'guest_email': row[9], 'num_guests': row[10],
                        'room_type': row[11], 'booking_reference': row[12]
                    }
                    
                    db_status = (booking_data['check_in_status'] or 'confirmed').lower()
                    status_mapping = {
                        'confirmed': 'pending', 'checked_in': 'checked-in',
                        'checked-in': 'checked-in', 'checked_out': 'checked-out',
                        'checked-out': 'checked-out', 'cancelled': 'checked-out'
                    }
                    frontend_status = status_mapping.get(db_status, 'pending')
                    
                    return {
                        "bookingId": booking_data['id'],
                        "hotelId": booking_data['hotel_id'],
                        "roomNumber": booking_data['room_number'],
                        "guestName": f"{booking_data['guest_first_name']} {booking_data['guest_last_name']}",
                        "hotelName": booking_data['hotel_name'],
                        "status": frontend_status,
                        "checkInDate": str(booking_data['check_in_date']) if booking_data['check_in_date'] else None,
                        "checkOutDate": str(booking_data['check_out_date']) if booking_data['check_out_date'] else None,
                        "guestEmail": booking_data['guest_email'],
                        "numGuests": booking_data['num_guests'],
                        "roomType": booking_data['room_type'],
                        "bookingReference": booking_data['booking_reference']
                    }
                else:
                    raise HTTPException(status_code=404, detail="Invalid booking reference. Please check and try again.")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in guest booking login: {e}")
        raise HTTPException(status_code=500, detail="Failed to process booking login")


@app.post("/api/guest/verify")
async def guest_verify(req: GuestVerifyRequest):
    """Legacy endpoint for guest verification with last name."""
    try:
        with get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute("""
                    SELECT id, room_number, guest_first_name, guest_last_name, check_out_date 
                    FROM bookings 
                    WHERE hotel_id = %s AND room_number = %s AND LOWER(guest_last_name) = LOWER(%s)
                """, (req.hotelId, req.roomNumber, req.lastName))
                
                row = cursor.fetchone()
                if row:
                    return {
                        "bookingId": row[0],
                        "roomNumber": row[1],
                        "guestName": f"{row[2]} {row[3]}",
                        "hotelId": req.hotelId
                    }
                else:
                    return {"error": "Invalid room or name for this hotel."}
    except Exception as e:
        logger.error(f"Error in guest verify: {e}")
        raise HTTPException(status_code=500, detail="Failed to verify guest")


@app.post("/api/guest/info")
async def get_guest_info(req: GuestInfoRequest):
    """Get guest information by hotel and room number."""
    try:
        with get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute("""
                    SELECT id, hotel_id, room_number, guest_first_name, guest_last_name, 
                           check_in_date, check_out_date, check_in_status, hotel_name,
                           guest_email, num_guests, room_type
                    FROM bookings 
                    WHERE hotel_id = %s AND room_number = %s
                    ORDER BY check_in_date DESC
                    LIMIT 1
                """, (req.hotelId, req.roomNumber))
                
                row = cursor.fetchone()
                if row:
                    db_status = (row[7] or 'confirmed').lower()
                    status_mapping = {
                        'confirmed': 'pending', 'checked_in': 'checked-in',
                        'checked-in': 'checked-in', 'checked_out': 'checked-out',
                        'checked-out': 'checked-out', 'cancelled': 'checked-out'
                    }
                    frontend_status = status_mapping.get(db_status, 'pending')
                    
                    return {
                        "bookingId": row[0], "hotelId": row[1], "roomNumber": row[2],
                        "guestName": f"{row[3]} {row[4]}", "hotelName": row[8],
                        "status": frontend_status,
                        "checkInDate": str(row[5]) if row[5] else None,
                        "checkOutDate": str(row[6]) if row[6] else None,
                        "guestEmail": row[9], "numGuests": row[10], "roomType": row[11]
                    }
                else:
                    raise HTTPException(status_code=404, detail="No booking found for this room. Please contact the front desk.")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting guest info: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch guest information")


@app.post("/api/chat/guest")
async def guest_chat(req: GuestChatRequest):
    """Concierge chat endpoint with RAG context and conversation history."""
    
    # 1. RETRIEVE HOTEL CONTEXT (RAG from Pinecone)
    hotel_doc = get_hotel_by_id(req.hotelId)
    base_context = hotel_doc.page_content if hotel_doc else "General hotel info."
    if hotel_doc:
        meta = hotel_doc.metadata
        hotel_context = (
            f"Hotel Name: {meta.get('hotel_name')}\n"
            f"Address: {meta.get('address')}\n"
            f"Star Rating: {meta.get('star_rating')}\n"
            f"Description & Amenities: {hotel_doc.page_content}\n"
        )
    else:
        hotel_context = "Hotel details currently unavailable."

    # 2. RETRIEVE MANAGER'S CUSTOM KNOWLEDGE
    custom_context = ""
    try:
        with get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute("SELECT context_text FROM hotel_knowledge WHERE hotel_id = %s", (int(req.hotelId),))
                result = cursor.fetchone()
                custom_context = result[0] if result else ""
    except Exception as e:
        logger.error(f"Error fetching hotel knowledge: {e}")
    
    full_context = f"""
    OFFICIAL HOTEL DATA:
    {base_context}
    
    MANAGER'S DAILY UPDATES (Highest Priority):
    {custom_context or "No specific daily updates."}
    """

    # 3. GET BOOKING REFERENCE if not provided
    booking_reference = req.bookingReference
    if not booking_reference:
        try:
            with get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute("SELECT booking_reference FROM bookings WHERE id = %s", (req.bookingId,))
                    result = cursor.fetchone()
                    booking_reference = result[0] if result else None
        except Exception as e:
            logger.error(f"Error fetching booking reference: {e}")
            booking_reference = None

    # 4. RETRIEVE CONVERSATION HISTORY (Last 10 messages for this booking_reference)
    history = []
    try:
        with get_connection() as conn:
            with conn.cursor() as cursor:
                # Fetch history based on booking_reference to maintain thread across sessions
                if booking_reference:
                    cursor.execute("""
                        SELECT gr.message_text, gr.bot_response 
                        FROM guest_requests gr
                        JOIN bookings b ON gr.booking_id = b.id
                        WHERE b.booking_reference = %s 
                        ORDER BY gr.created_at ASC 
                        LIMIT 10
                    """, (booking_reference,))
                else:
                    # Fallback to booking_id if no reference available
                    cursor.execute("""
                        SELECT message_text, bot_response 
                        FROM guest_requests 
                        WHERE booking_id = %s 
                        ORDER BY created_at ASC 
                        LIMIT 10
                    """, (req.bookingId,))
                
                rows = cursor.fetchall()
                for row in rows:
                    history.append({"role": "user", "content": row[0]})
                    history.append({"role": "assistant", "content": row[1]})
    except Exception as e:
        logger.error(f"Error fetching conversation history: {e}")

    # 5. PROCESS WITH AGENT
    agent_result = await process_guest_message(
        message=req.message, 
        guest_name=req.guestName, 
        room_number=req.roomNumber,
        hotel_context=full_context,
        history=history
    )
    
    # 6. FILTER: STOP LOGGING "OTHER" REQUESTS
    if agent_result['request_type'] == 'other':
        return {
            "response": agent_result['response'],
            "requestId": None,
            "type": "other",
            "status": "ignored"
        }

    # 7. LOG ACTIONABLE REQUESTS (stores with booking_id, but history is fetched by booking_reference)
    try:
        with get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute("""
                    INSERT INTO guest_requests 
                    (booking_id, room_number, guest_name, message_text, bot_response, 
                     request_type, is_service_request, status, priority)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                    RETURNING id
                """, (req.bookingId, req.roomNumber, req.guestName, 
                      req.message, agent_result['response'], 
                      agent_result['request_type'],
                      agent_result['is_service_request'],
                      'pending', agent_result['priority']))
                
                request_id = cursor.fetchone()[0]
                
                return {
                    "response": agent_result['response'],
                    "requestId": request_id,
                    "type": agent_result['request_type'],
                    "status": "pending"
                }
    except Exception as e:
        logger.error(f"Error logging guest request: {e}")
        return {"response": agent_result['response'], "error": "Logged locally (DB unavailable)"}


# ======================================================
# ADMIN API ROUTES
# ======================================================

@app.get("/api/admin/requests")
async def get_admin_requests(hotel_id: Optional[int] = None, category: Optional[str] = None):
    """Fetch guest requests sorted by priority."""
    try:
        with get_connection() as conn:
            with conn.cursor() as cursor:
                query = """
                    SELECT r.* FROM guest_requests r
                    JOIN bookings b ON r.booking_id = b.id
                    WHERE r.request_type != 'other' AND r.status != 'ignored'
                """
                params = []
                
                if hotel_id:
                    query += " AND b.hotel_id = %s"
                    params.append(hotel_id)
                
                if category and category != 'all':
                    query += " AND r.request_type = %s"
                    params.append(category)
                
                query += """
                    ORDER BY 
                    CASE 
                        WHEN r.priority = 'high' THEN 1 
                        WHEN r.priority = 'medium' THEN 2 
                        ELSE 3 
                    END, 
                    r.created_at DESC 
                    LIMIT 100
                """
                
                cursor.execute(query, tuple(params))
                rows = cursor.fetchall()
                
                # Get column names
                columns = [desc[0] for desc in cursor.description]
                
                # Convert to list of dicts
                return [dict(zip(columns, row)) for row in rows]
    except Exception as e:
        logger.error(f"Error fetching admin requests: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch requests")


@app.patch("/api/admin/requests/{request_id}")
async def update_request(request_id: int, update: RequestUpdate):
    """Update guest request status."""
    try:
        with get_connection() as conn:
            with conn.cursor() as cursor:
                resolved_at = datetime.now() if update.status == 'resolved' else None
                cursor.execute(
                    "UPDATE guest_requests SET status = %s, assigned_to = %s, resolved_at = %s WHERE id = %s",
                    (update.status, update.assigned_to, resolved_at, request_id)
                )
        return {"status": "success", "id": request_id}
    except Exception as e:
        logger.error(f"Error updating request: {e}")
        raise HTTPException(status_code=500, detail="Failed to update request")


@app.get("/api/admin/stats")
async def get_admin_stats():
    """Fetch statistics for the Admin Dashboard."""
    try:
        with get_connection() as conn:
            with conn.cursor() as cursor:
                # Guest requests stats
                cursor.execute("SELECT status, COUNT(*) as count FROM guest_requests GROUP BY status")
                request_rows = cursor.fetchall()
                request_stats = {row[0]: row[1] for row in request_rows}
                
                # Count confirmed bookings (status = 'confirmed')
                cursor.execute("SELECT COUNT(*) FROM bookings WHERE status = 'confirmed'")
                confirmed_bookings = cursor.fetchone()[0]
                
                # Count checked in (check_in_status = 'checked_in')
                cursor.execute("SELECT COUNT(*) FROM bookings WHERE check_in_status = 'checked_in'")
                checked_in = cursor.fetchone()[0]
                
                # Count checked out today (check_in_status = 'checked_out' AND check_out_date = TODAY)
                cursor.execute("""
                    SELECT COUNT(*) 
                    FROM bookings 
                    WHERE check_in_status = 'checked_out' 
                    AND DATE(check_out_date) = CURRENT_DATE
                """)
                checked_out_today = cursor.fetchone()[0]
                
                return {
                    "pending_requests": request_stats.get("pending", 0),
                    "resolved_requests": request_stats.get("resolved", 0),
                    "total_requests": sum(request_stats.values()),
                    "confirmed_bookings": confirmed_bookings,
                    "checked_in": checked_in,
                    "checked_out_today": checked_out_today
                }
    except Exception as e:
        logger.error(f"Error fetching stats: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch statistics")
    except Exception as e:
        logger.error(f"Error fetching stats: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch statistics")


@app.get("/api/admin/bookings")
async def get_admin_bookings(hotel_id: Optional[int] = None):
    """Fetch all bookings for a hotel with guest details."""
    try:
        with get_connection() as conn:
            with conn.cursor() as cursor:
                query = """
                    SELECT 
                        id, hotel_id, booking_reference, room_number,
                        guest_first_name || ' ' || guest_last_name as guest_name,
                        guest_email, check_in_date, check_out_date, num_guests,
                        status, room_type, check_in_status, hotel_name, created_at
                    FROM bookings
                """
                
                params = []
                if hotel_id:
                    query += " WHERE hotel_id = %s"
                    params.append(hotel_id)
                
                query += """
                    ORDER BY 
                        CASE COALESCE(check_in_status, 'pending')
                            WHEN 'checked_in' THEN 0
                            WHEN 'pending' THEN 1
                            WHEN 'checked_out' THEN 2
                            ELSE 3
                        END,
                        check_in_date DESC
                    LIMIT 100
                """
                
                cursor.execute(query, tuple(params))
                rows = cursor.fetchall()
                columns = [desc[0] for desc in cursor.description]
                
                result = []
                for row in rows:
                    booking = dict(zip(columns, row))
                    if not booking.get('check_in_status'):
                        booking['check_in_status'] = 'confirmed'
                    booking['status'] = booking['check_in_status']
                    result.append(booking)
                
                return result
    except Exception as e:
        logger.error(f"Error fetching bookings: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch bookings")


@app.patch("/api/admin/bookings/{booking_id}/status")
async def update_booking_status(booking_id: int, status_update: dict):
    """Admin updates booking check-in status."""
    try:
        new_status = status_update.get('status')
        valid_statuses = ['pending', 'checked_in', 'checked_out']  # FIXED: Only valid check_in_status values
        
        if new_status not in valid_statuses:
            raise HTTPException(status_code=400, detail=f"Invalid status. Must be one of: {valid_statuses}")
        
        with get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute("SELECT id FROM bookings WHERE id = %s", (booking_id,))
                if not cursor.fetchone():
                    raise HTTPException(status_code=404, detail="Booking not found")
                
                cursor.execute("UPDATE bookings SET check_in_status = %s WHERE id = %s", (new_status, booking_id))
        
        logger.info(f"Booking status updated: {booking_id} -> {new_status}")
        
        return {
            "success": True,
            "message": f"Booking status updated to {new_status}",
            "booking_id": booking_id,
            "new_status": new_status
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating booking status: {e}")
        raise HTTPException(status_code=500, detail="Failed to update booking status")


@app.post("/api/admin/upload-knowledge")
async def upload_knowledge(hotel_id: int, file: UploadFile = File(...)):
    """Manager uploads a text file for hotel knowledge base."""
    try:
        content = await file.read()
        text_content = content.decode("utf-8")
        
        with get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute("""
                    INSERT INTO hotel_knowledge (hotel_id, context_text, updated_at)
                    VALUES (%s, %s, NOW())
                    ON CONFLICT (hotel_id) 
                    DO UPDATE SET context_text = %s, updated_at = NOW()
                """, (hotel_id, text_content, text_content))
        
        return {"status": "success", "message": "Knowledge base updated"}
    except Exception as e:
        logger.error(f"Error uploading knowledge: {e}")
        raise HTTPException(status_code=500, detail="Failed to upload knowledge")


@app.get("/api/admin/knowledge/{hotel_id}")
async def get_knowledge(hotel_id: int):
    """Get current knowledge base for a hotel."""
    try:
        with get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute("SELECT context_text FROM hotel_knowledge WHERE hotel_id = %s", (hotel_id,))
                result = cursor.fetchone()
                return {"context": result[0] if result else ""}
    except Exception as e:
        logger.error(f"Error fetching knowledge: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch knowledge")


# ======================================================
# UTILITY ROUTES
# ======================================================

@app.get("/api/v1/dbtest")
async def test_db_connection():
    """Test database connection and list tables."""
    try:
       tables = list_tables()
       return {
            "status": "success",
            "tables": tables,
            "count": len(tables)
        }
    except Exception as e:
        logger.error("Error testing DB", error=str(e))
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to test DB: {str(e)}")


# ======================================================
# LOCAL RUN ENTRYPOINT
# ======================================================

if __name__ == "__main__":
    """Run the server directly."""
    import uvicorn
    port = int(os.environ.get("PORT", 8000)) 
    uvicorn.run(app, host="0.0.0.0", port=port)