"""
HotelIQ Backend Server
======================

FastAPI application for the HotelIQ hotel comparison and booking chatbot.
"""

import os
import uuid
import logging
from pathlib import Path
from typing import List, Optional
from dataclasses import dataclass

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

from agents import agent_graph
from agents.state import HotelIQState
import bucket_util
import path as path_util

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ======================================================
# GCP CREDENTIALS SETUP
# ======================================================

# Set GCP credentials path if not already set
if not os.getenv('GOOGLE_APPLICATION_CREDENTIALS'):
    credentials_path = Path(__file__).parent / "config" / "gcp-service-account.json"
    if credentials_path.exists():
        os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = str(credentials_path)
        logger.info(f"✅ GCP credentials loaded from: {credentials_path}")
    else:
        logger.warning(f"⚠️ GCP credentials file not found at: {credentials_path}")

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
    logger.info(f"🔍 Checking data files for city: {city}")
    
    # Check which files exist
    files_status = check_data_files_exist(city)
    all_exist = all(files_status.values())
    
    if all_exist:
        logger.info(f"✅ All data files already exist locally. Skipping download.")
        logger.info(f"   Location: {Path(path_util.get_processed_dir(city))}")
        for file_name in files_status.keys():
            logger.info(f"   ✓ {file_name}.csv")
        return True
    
    # Log which files need to be downloaded
    missing_files = [name for name, exists in files_status.items() if not exists]
    logger.info(f"🔄 Downloading missing data files: {', '.join(missing_files)}")
    
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
                logger.info(f"⏭️  Skipping {file_name}.csv (already exists)")
                skipped_count += 1
                success_count += 1
                continue
            
            logger.info(f"📥 Downloading {file_name}.csv...")
            bucket_util.download_file_from_gcs(gcs_path, local_path)
            logger.info(f"✅ Successfully downloaded {file_name}.csv")
            success_count += 1
        except Exception as e:
            logger.error(f"❌ Failed to download {file_name}.csv: {e}")
            # Continue downloading other files even if one fails
    
    if skipped_count > 0:
        logger.info(f"📦 Downloaded {success_count - skipped_count}/{len(files_to_download)} files, skipped {skipped_count} existing files")
    else:
        logger.info(f"📦 Downloaded {success_count}/{len(files_to_download)} files successfully")
    
    return success_count == len(files_to_download)

# ======================================================
# FASTAPI APP SETUP
# ======================================================

BASE_DIR = Path(__file__).resolve().parent.parent
FRONTEND_DIR = BASE_DIR / "frontend"

app = FastAPI(title="HotelIQ Comparison API")

# Add startup event to download data
@app.on_event("startup")
async def startup_event():
    """Execute on application startup."""
    logger.info("🚀 Starting HotelIQ API...")
    download_processed_data()
    logger.info("✅ Startup complete!")

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve frontend static files
if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")
else:
    print(f"⚠️ Frontend directory not found: {FRONTEND_DIR}")

# Route to return chat.html UI
@app.get("/chat")
async def chat_page():
    """Serve the chat interface HTML page."""
    return FileResponse(FRONTEND_DIR / "chat.html")


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
                    print(f"📜 Retrieved {len(messages)} previous messages for thread {thread_id}")
        except Exception as e:
            print(f"⚠️ Could not retrieve previous state: {e}")
        
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
print("✅ ChatService ready.")


# ======================================================
# API ROUTES
# ======================================================

class ChatRequest(BaseModel):
    """Request model for chat endpoint."""
    message: str
    user_id: str
    hotel_id: str  # Required hotel_id from frontend
    thread_id: Optional[str] = None


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
    """
    res = await chat_service.process_message(
        message=request.message,
        thread_id=request.thread_id,
        user_id=request.user_id,
        hotel_id=request.hotel_id,
    )
    return ChatResponseModel(
        response=res.response,
        thread_id=res.thread_id,
        followup_suggestions=res.followup_suggestions,
    )


print("✅ FastAPI app created.")


# ======================================================
# LOCAL RUN ENTRYPOINT
# ======================================================

if __name__ == "__main__":
    """
    Run the server directly.
    
    Preferred method:
        cd backend
        uvicorn main:app --reload --port 8001
    """
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)

