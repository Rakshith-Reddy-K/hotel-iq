"""
Configuration and Global State
===============================

Contains all configuration, paths, and global state variables for the HotelIQ system.
"""

import os
import sys
from pathlib import Path
from typing import Any, Dict, List
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings

# Add parent directory to path to import path utility
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import path as path_util

# ======================================================
# PATHS
# ======================================================

# BASE_DIR = "Model Development" (one level above backend)
BASE_DIR = Path(__file__).resolve().parent.parent.parent
FRONTEND_DIR = BASE_DIR / "frontend"
BOOKINGS_PATH = BASE_DIR / "booking_requests.json"

# Get city from environment variable, default to 'boston'
CITY = os.getenv('CITY', 'boston')

# Data paths - now dynamically set based on CITY environment variable
HOTELS_PATH = Path(path_util.get_processed_hotels_path(CITY))
REVIEWS_PATH = Path(path_util.get_processed_reviews_path(CITY))
AMENITIES_PATH = Path(path_util.get_processed_amenities_path(CITY))
POLICIES_PATH = Path(path_util.get_processed_policies_path(CITY))

# ======================================================
# GLOBAL STATE
# ======================================================

# Track last recommended hotels per thread (for "the first one", "second one")
last_suggestions: Dict[str, List[Dict[str, str]]] = {}

# Track conversation context per thread
conversation_context: Dict[str, Dict[str, Any]] = {}
# Structure: {
#     thread_id: {
#         "questions": [list of user questions],
#         "hotels_discussed": [list of hotel names in order discussed],
#         "current_hotel": "name of most recently discussed hotel",
#         "question_hotel_map": [(question, hotel_name), ...],
#         "explicit_hotel_mentioned": None,  # Track if user explicitly named a hotel
#         "assistant_snippets": [list of first 60 words from assistant responses],
#         "conversation_pairs": [(user_msg, assistant_snippet), ...]  # Track Q&A pairs
#     }
# }

# Fake "database" for bookings (in memory + JSON file)
bookings_log: List[Dict[str, Any]] = []

# ======================================================
# LLM & EMBEDDINGS
# ======================================================

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not found in .env file.")

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.0,
    openai_api_key=OPENAI_API_KEY,
)

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

print("✅ LLM and embeddings configured.")

