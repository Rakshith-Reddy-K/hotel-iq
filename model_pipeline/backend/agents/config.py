"""
Configuration
"""

import os
import sys
from pathlib import Path
from typing import Any, Dict, List
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from logger_config import get_logger

logger = get_logger(__name__)


sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import path as path_util




BASE_DIR = Path(__file__).resolve().parent.parent.parent
FRONTEND_DIR = BASE_DIR / "frontend"
BOOKINGS_PATH = BASE_DIR / "booking_requests.json"


CITY = os.getenv('CITY', 'boston')


HOTELS_PATH = Path(path_util.get_processed_hotels_path(CITY))
REVIEWS_PATH = Path(path_util.get_processed_reviews_path(CITY))
AMENITIES_PATH = Path(path_util.get_processed_amenities_path(CITY))
POLICIES_PATH = Path(path_util.get_processed_policies_path(CITY))

bookings_log: List[Dict[str, Any]] = []



load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not found in .env file.")

from langfuse.langchain import CallbackHandler

langfuse_handler = CallbackHandler()

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.0,
    openai_api_key=OPENAI_API_KEY,
    callbacks=[langfuse_handler],
)


logger.info("LLM configured.")


