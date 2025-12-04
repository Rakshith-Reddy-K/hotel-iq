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

import asyncpg
import uuid
import logging
from pathlib import Path
from typing import List, Optional
from dataclasses import dataclass

from fastapi import FastAPI, HTTPException, Depends 
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from contextlib import asynccontextmanager

from agents import agent_graph
from agents.state import HotelIQState
from agents.validation import sanitize_user_input
import bucket_util
import path as path_util

from agents.concierge_agent import process_guest_message
from logger_config import configure_logger, get_logger
import json
from datetime import datetime
from fastapi import HTTPException, status
from typing import Dict, Any
from sql.queries import list_tables
from auth_routes import auth_router
from hotel_routes import hotel_router

# Setup logging
configure_logger()
logger = get_logger(__name__)

# ======================================================
# GCP CREDENTIALS SETUP
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

# Database Config (For Concierge/Admin)
DB_DSN = f"postgresql://{os.getenv('CLOUD_DB_USER')}:{os.getenv('CLOUD_DB_PASSWORD')}@{os.getenv('CLOUD_DB_HOST')}:{os.getenv('CLOUD_DB_PORT')}/{os.getenv('CLOUD_DB_NAME')}"

BASE_DIR = Path(__file__).resolve().parent.parent
FRONTEND_DIR = BASE_DIR / "frontend"

# ======================================================
# DATA INITIALIZATION
# ======================================================

def check_data_files_exist(city: str) -> dict:
    """
    Check if all required data files exist locally.
    
    Args:
        city: City name for which to check data files
        
    Returns:
        Dictionary with file names as keys and existence status as values
    """
    files_status = {
        'hotels': Path(path_util.get_processed_hotels_path(city)).exists(),
        'amenities': Path(path_util.get_processed_amenities_path(city)).exists(),
        'policies': Path(path_util.get_processed_policies_path(city)).exists(),
        'reviews': Path(path_util.get_processed_reviews_path(city)).exists(),
    }
    return files_status


def download_processed_data():
    """
    Download processed CSV files from GCS to local data folder on startup.
    Only downloads files that don't already exist locally.
    Downloads: hotels.csv, amenities.csv, policies.csv, reviews.csv
    """
    # Get city from environment variable, default to 'boston'
    city = os.getenv('CITY', 'boston')
    logger.info("Checking data files", city=city)
    
    # Check which files exist
    files_status = check_data_files_exist(city)
    all_exist = all(files_status.values())
    
    if all_exist:
        logger.info("All data files already exist locally. Skipping download.", location=str(Path(path_util.get_processed_dir(city))))
        for file_name in files_status.keys():
            logger.info(f"   ✓ {file_name}.csv")
        return True
    
    # Log which files need to be downloaded
    missing_files = [name for name, exists in files_status.items() if not exists]
    logger.info("Downloading missing data files", missing_files=missing_files)
    
    # Define the files to download
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
    
    # Download only missing files
    success_count = 0
    skipped_count = 0
    for file_name, gcs_path, local_path in files_to_download:
        try:
            # Check if file already exists
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
            # Continue downloading other files even if one fails
    
    if skipped_count > 0:
        logger.info("Download summary", downloaded=success_count - skipped_count, total=len(files_to_download), skipped=skipped_count)
    else:
        logger.info("Download summary", downloaded=success_count, total=len(files_to_download))
    
    return success_count == len(files_to_download)

# ======================================================
# FASTAPI APP SETUP
# ======================================================

BASE_DIR = Path(__file__).resolve().parent.parent
FRONTEND_DIR = BASE_DIR / "frontend"

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
    """
    Readiness check - verifies app is ready to serve traffic.
    Checks if data files are available.
    """
    city = os.getenv('CITY', 'boston')
    files_status = check_data_files_exist(city)
    all_ready = all(files_status.values())
    
    return {
        "status": "ready" if all_ready else "not_ready",
        "service": "hoteliq-backend",
        "data_files": files_status,
        "city": city
    }

# Add startup event to download data
@app.on_event("startup")
async def startup_event():
    """Execute on application startup: Download data & Connect to DB."""
    logger.info("Starting HotelIQ API...")
    download_processed_data()
    try:
        app.state.pool = await asyncpg.create_pool(DB_DSN)
        logger.info(" Connected to Database (AsyncPG)")
    except Exception as e:
        logger.error(f" DB Connection Failed: {e}")
        app.state.pool = None
    logger.info("Startup complete!")

@app.on_event("shutdown")
async def shutdown_event():
    """Close database connection on shutdown."""
    if hasattr(app.state, 'pool') and app.state.pool:
        await app.state.pool.close()
        logger.info("🛑 Database connection closed")

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
# Serve frontend static files
if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")
else:
    logger.warning("Frontend directory not found", path=str(FRONTEND_DIR))

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
        """
        Process a user message through the agent graph.
        
        Args:
            message: User's message
            thread_id: Optional thread ID for conversation continuity
            user_id: User identifier
            hotel_id: Hotel ID for context-specific queries
            
        Returns:
            ChatServiceResponse with assistant's response and metadata
        """
        # Generate thread ID if not provided
        if not thread_id:
            thread_id = f"thread_{user_id}_{uuid.uuid4()}"

        # Retrieve previous state to get conversation history
        messages = []
        try:
            config = {"configurable": {"thread_id": thread_id}}
            previous_state = self.agent_graph.get_state(config)
            
            # If we have previous state, get the message history
            if previous_state and previous_state.values:
                previous_messages = previous_state.values.get("messages", [])
                if previous_messages:
                    messages = previous_messages.copy()
                    logger.info("Retrieved previous messages", count=len(messages), thread_id=thread_id)
        except Exception as e:
            logger.warning("Could not retrieve previous state", error=str(e))
        
        # Append new user message
        messages.append({"role": "user", "content": message})

        # Initialize state with full conversation history
        init_state: HotelIQState = {
            "messages": messages,
            "user_id": user_id,
            "thread_id": thread_id,
            "hotel_id": hotel_id,
        }

        # Run through agent graph
        result_state: HotelIQState = await self.agent_graph.ainvoke(
            init_state,
            config={"configurable": {"thread_id": thread_id}},
        )

        # Extract assistant response
        assistant_msg = result_state["messages"][-1]["content"]

        # Generate follow-up suggestions
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


# Initialize chat service
chat_service = ChatService(agent_graph)
logger.info("ChatService ready.")


# ======================================================
# API ROUTES
# ======================================================
class GuestVerifyRequest(BaseModel):
    roomNumber: str
    lastName: str

class GuestChatRequest(BaseModel):
    message: str
    roomNumber: str
    guestName: str
    bookingId: int

class RequestUpdate(BaseModel):
    status: str
    assigned_to: Optional[str] = None

class ChatRequest(BaseModel):
    """Request model for chat endpoint with validation."""
    message: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="User message (1-2000 characters)"
    )
    user_id: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="User identifier"
    )
    hotel_id: str = Field(
        ...,
        min_length=1,
        max_length=50,
        description="Hotel identifier"
    )
    thread_id: Optional[str] = Field(
        None,
        max_length=200,
        description="Thread identifier for conversation continuity"
    )


class ChatResponseModel(BaseModel):
    """Response model for chat endpoint."""
    response: str
    thread_id: str
    followup_suggestions: List[str]


@app.post("/api/v1/chat/message", response_model=ChatResponseModel)
async def send_message(request: ChatRequest):
    """
    Process a chat message.
    
    This endpoint receives user messages and returns AI-generated responses
    along with follow-up suggestions.
    
    Input validation and sanitization is performed automatically by Pydantic.
    """
    # Additional sanitization for extra security
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


# --- CONCIERGE: GUEST VERIFICATION ---
@app.post("/api/guest/verify")
async def guest_verify(req: GuestVerifyRequest):
    """Verify guest credentials against bookings table."""
    if not hasattr(app.state, 'pool') or not app.state.pool:
        raise HTTPException(status_code=503, detail="Database not ready")

    async with app.state.pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            SELECT id, room_number, guest_first_name, guest_last_name, check_out_date 
            FROM bookings 
            WHERE room_number = $1 AND LOWER(guest_last_name) = LOWER($2)
            """, 
            req.roomNumber, req.lastName
        )

        if row:
            return {
                "bookingId": row['id'],
                "roomNumber": row['room_number'],
                "guestName": f"{row['guest_first_name']} {row['guest_last_name']}",
                "checkoutDate": str(row['check_out_date'])
            }
        else:
            return {"error": "Invalid room number or last name."}

# --- CONCIERGE: GUEST CHAT ---
@app.post("/api/chat/guest")
async def guest_chat(req: GuestChatRequest):
    """Concierge chat for checked-in guests."""
    # 1. Process with Concierge Agent
    agent_result = await process_guest_message(req.message, req.guestName, req.roomNumber)
    
    # 2. Log to Database
    if hasattr(app.state, 'pool') and app.state.pool:
        async with app.state.pool.acquire() as conn:
            request_id = await conn.fetchval(
                """
                INSERT INTO guest_requests 
                (booking_id, room_number, guest_name, message_text, bot_response, request_type, status, priority)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                RETURNING id
                """,
                req.bookingId, req.roomNumber, req.guestName, 
                req.message, agent_result['response'], 
                agent_result['request_type'],
                'pending' if agent_result['is_service_request'] else 'resolved', 
                agent_result['priority']
            )
            
            return {
                "response": agent_result['response'],
                "requestId": request_id,
                "type": agent_result['request_type'],
                "status": "pending" if agent_result['is_service_request'] else "resolved"
            }
    
    return {"response": agent_result['response'], "error": "Logged locally (DB unavailable)"}

# --- ADMIN: FETCH REQUESTS ---
@app.get("/api/admin/requests")
async def get_admin_requests(status: Optional[str] = None):
    """Fetch requests for the Admin Dashboard."""
    if not hasattr(app.state, 'pool') or not app.state.pool:
        raise HTTPException(status_code=503, detail="Database not ready")
        
    async with app.state.pool.acquire() as conn:
        if status:
            rows = await conn.fetch("SELECT * FROM guest_requests WHERE status = $1 ORDER BY created_at DESC", status)
        else:
            rows = await conn.fetch("SELECT * FROM guest_requests ORDER BY created_at DESC LIMIT 50")
        return [dict(row) for row in rows]

# --- ADMIN: UPDATE REQUEST ---
@app.patch("/api/admin/requests/{request_id}")
async def update_request(request_id: int, update: RequestUpdate):
    """Update request status (e.g., mark as resolved)."""
    if not hasattr(app.state, 'pool') or not app.state.pool:
        raise HTTPException(status_code=503, detail="Database not ready")

    from datetime import datetime
    async with app.state.pool.acquire() as conn:
        await conn.execute(
            "UPDATE guest_requests SET status = $1, assigned_to = $2, resolved_at = $3 WHERE id = $4",
            update.status, update.assigned_to, 
            datetime.now() if update.status == 'resolved' else None,
            request_id
        )
    return {"status": "success", "id": request_id}

@app.get("/api/admin/stats")
async def get_admin_stats():
    """Fetch statistics for the Admin Dashboard."""
    if not hasattr(app.state, 'pool') or not app.state.pool:
        raise HTTPException(status_code=503, detail="Database not ready")

    async with app.state.pool.acquire() as conn:
        # Get total pending vs resolved
        status_counts = await conn.fetch(
            "SELECT status, COUNT(*) as count FROM guest_requests GROUP BY status"
        )
        
        # Get average response time (mock logic for now as we need 'resolved_at' populated)
        avg_time = await conn.fetchval(
            "SELECT AVG(EXTRACT(EPOCH FROM (resolved_at - created_at))/60)::int FROM guest_requests WHERE status = 'resolved'"
        )

        stats = {row['status']: row['count'] for row in status_counts}
        
        return {
            "pending_requests": stats.get("pending", 0),
            "resolved_requests": stats.get("resolved", 0),
            "avg_response_time": avg_time or 0,
            "total_requests": sum(stats.values())
        }

logger.info("FastAPI app created.")

@app.post("/api/v1/chat/save")
async def save_conversation(request: SaveConversationRequest):
    """
    Save chat conversation to Google Cloud Storage.
    
    Stores the conversation as JSON in GCS with the structure:
    conversations/{hotelId}/{userId}/{threadId}.json
    """
    try:
        # Sanitize inputs
        thread_id = sanitize_user_input(request.threadId)
        user_id = sanitize_user_input(request.userId)
        hotel_id = sanitize_user_input(request.hotelId)
        
        # Prepare conversation data
        conversation_data = {
            "threadId": thread_id,
            "userId": user_id,
            "hotelId": hotel_id,
            "messages": [msg.dict(by_alias=True) for msg in request.messages],
            "lastUpdated": datetime.utcnow().isoformat(),
        }
        
        # Create blob path
        blob_name = f"conversations/{hotel_id}/{user_id}/{thread_id}.json"
        
        # Save to temporary file
        temp_dir = Path("./temp")
        temp_dir.mkdir(exist_ok=True)
        temp_file = temp_dir / f"{thread_id}.json"
        
        with open(temp_file, 'w') as f:
            json.dump(conversation_data, f, indent=2)
        
        # Upload to GCS
        bucket_util.upload_file_to_gcs(str(temp_file), blob_name)
        
        # Clean up temp file
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
async def get_conversation(
    thread_id: str,
    userId: str,
    hotelId: str
):
    """
    Retrieve a chat conversation from Google Cloud Storage.
    
    Args:
        thread_id: Thread identifier
        userId: User identifier (query param)
        hotelId: Hotel identifier (query param)
    
    Returns:
        Conversation data including all messages
    """
    try:
        # Sanitize inputs
        thread_id = sanitize_user_input(thread_id)
        user_id = sanitize_user_input(userId)
        hotel_id = sanitize_user_input(hotelId)
        
        # Create blob path
        blob_name = f"conversations/{hotel_id}/{user_id}/{thread_id}.json"
        
        # Download from GCS
        temp_dir = Path("./temp")
        temp_dir.mkdir(exist_ok=True)
        temp_file = temp_dir / f"{thread_id}_download.json"
        
        try:
            bucket_util.download_file_from_gcs(blob_name, str(temp_file))
        except FileNotFoundError:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Conversation not found"
            )
        
        # Read conversation data
        with open(temp_file, 'r') as f:
            conversation_data = json.load(f)
        
        # Clean up temp file
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
async def delete_conversation(
    thread_id: str,
    request: DeleteConversationRequest
):
    """
    Delete a chat conversation from Google Cloud Storage.
    
    Args:
        thread_id: Thread identifier
        request: Contains userId and hotelId
    
    Returns:
        Success status
    """
    try:
        # Sanitize inputs
        thread_id = sanitize_user_input(thread_id)
        user_id = sanitize_user_input(request.userId)
        hotel_id = sanitize_user_input(request.hotelId)
        
        # Create blob path
        blob_name = f"conversations/{hotel_id}/{user_id}/{thread_id}.json"
        
        # Delete from GCS
        bucket = bucket_util.get_bucket()
        blob = bucket.blob(blob_name)
        
        if not blob.exists():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Conversation not found"
            )
        
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
async def list_conversations(
    userId: str,
    hotelId: str,
    limit: int = 50
):
    """
    List all conversations for a user at a specific hotel.
    
    Args:
        userId: User identifier
        hotelId: Hotel identifier
        limit: Maximum number of conversations to return (default: 50)
    
    Returns:
        List of conversation metadata
    """
    try:
        # Sanitize inputs
        user_id = sanitize_user_input(userId)
        hotel_id = sanitize_user_input(hotelId)
        
        # Create prefix for listing
        prefix = f"conversations/{hotel_id}/{user_id}/"
        
        # List blobs
        bucket = bucket_util.get_bucket()
        blobs = list(bucket.list_blobs(prefix=prefix, max_results=limit))
        
        conversations = []
        for blob in blobs:
            # Extract thread_id from blob name
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


@app.get("/api/v1/dbtest")
async def test_db_connection():
    try:
       tables = list_tables()
       return {
            "status": "success",
            "tables": tables,
            "count": len(tables)
        }
    except Exception as e:
        logger.error("Error listing conversations", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list conversations: {str(e)}"
        )

# ======================================================
# LOCAL RUN ENTRYPOINT
# ======================================================

if __name__ == "__main__":
    """
    Run the server directly.
    
    Preferred method:
        cd backend
        uvicorn main:app --reload --port 8000
    """
    import uvicorn
    port = int(os.environ.get("PORT", 8000)) 
    uvicorn.run(app, host="0.0.0.0", port=port)

