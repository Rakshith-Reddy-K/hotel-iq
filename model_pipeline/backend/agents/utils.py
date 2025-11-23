"""
Utility Functions and Data Loading
===================================

Contains all helper functions, retrievers, data loading, and prompts.
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, Iterable, List, Optional
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.chat_history import InMemoryChatMessageHistory

from .config import (
    llm, HOTELS_PATH, REVIEWS_PATH
)
from .pinecone_retrieval import embeddings
from .prompt_loader import get_prompts
from logger_config import get_logger

logger = get_logger(__name__)




# Global variables for lazy loading
_data_loaded = False
hotels_df = None
df = None
hotel_info_docs: List[Document] = []
review_docs: List[Document] = []
hotel_info_retriever = None
reviews_retriever = None


def _load_data():
    """
    Lazy load hotel and review data from CSV files.
    This is called on first access to avoid loading at import time.
    """
    global _data_loaded, hotels_df, df, hotel_info_docs, review_docs, hotel_info_retriever, reviews_retriever
    
    if _data_loaded:
        return
    
    logger.info("Loading hotel data from CSV files...")
    
    # Load hotels data
    hotels_df = pd.read_csv(HOTELS_PATH).fillna("")

    possible_name_cols = ["official_name", "hotel_name", "name"]
    name_col = next((c for c in possible_name_cols if c in hotels_df.columns), None)
    if name_col is None:
        hotels_df["official_name"] = ""
    else:
        hotels_df["official_name"] = hotels_df[name_col].astype(str)

    if "hotel_id" not in hotels_df.columns:
        raise ValueError("Hotels CSV file must contain a 'hotel_id' column.")

    for col in ["description", "additional_info", "address"]:
        if col not in hotels_df.columns:
            hotels_df[col] = ""

    if "star_rating" not in hotels_df.columns:
        hotels_df["star_rating"] = ""

    hotels_df["hotel_info_text"] = (
        hotels_df["description"].astype(str)
        + "\n\n"
        + hotels_df["additional_info"].astype(str)
        + "\n\nAddress: "
        + hotels_df["address"].astype(str)
    ).str.strip()

    # Load reviews data
    if REVIEWS_PATH.exists():
        raw_reviews_df = pd.read_csv(REVIEWS_PATH).fillna("")
        if "hotel_id" not in raw_reviews_df.columns:
            raise ValueError("Reviews CSV file must contain 'hotel_id'.")

        text_cols = [c for c in raw_reviews_df.columns if c != "hotel_id"]
        if not text_cols:
            raise ValueError("Reviews CSV must have at least one text column besides 'hotel_id'.")

        def row_to_text(row):
            vals = [str(row[c]) for c in text_cols if str(row[c]).strip()]
            return " ".join(vals)

        raw_reviews_df["__review_text__"] = raw_reviews_df.apply(row_to_text, axis=1)
        aggs = raw_reviews_df.groupby("hotel_id")["__review_text__"].apply(
            lambda x: "\n\n".join(v for v in x if str(v).strip())
        ).reset_index()
        aggs = aggs.rename(columns={"__review_text__": "reviews_text"})
        reviews_df = aggs
    else:
        logger.warning("Reviews CSV file not found; continuing without reviews.")
        reviews_df = pd.DataFrame(columns=["hotel_id", "reviews_text"])

    df = hotels_df.merge(reviews_df, on="hotel_id", how="left")
    df["reviews_text"] = df["reviews_text"].fillna("")

    logger.info("Hotels dataframe ready", sample=df[["hotel_id", "official_name", "star_rating"]].head().to_dict())

    # Build document collections
    hotel_info_docs.clear()
    review_docs.clear()

    for _, row in df.iterrows():
        meta = {
            "hotel_id": str(row["hotel_id"]),
            "name": str(row["official_name"]),
            "star_rating": str(row["star_rating"]),
        }

        info_text = str(row["hotel_info_text"]).strip()
        if info_text:
            hotel_info_docs.append(
                Document(
                    page_content=info_text,
                    metadata={**meta, "source": "hotel_info"},
                )
            )

        reviews_text = str(row["reviews_text"]).strip()
        if reviews_text:
            review_docs.append(
                Document(
                    page_content=reviews_text,
                    metadata={**meta, "source": "review"},
                )
            )

    logger.info(
        "Prepared docs",
        hotel_info_docs=len(hotel_info_docs),
        review_docs=len(review_docs)
    )

    # Initialize retrievers
    hotel_info_retriever = SimpleInMemoryRetriever(hotel_info_docs, embeddings, k=8)
    reviews_retriever = (
        SimpleInMemoryRetriever(review_docs, embeddings, k=8)
        if review_docs
        else None
    )

    logger.info("SimpleInMemoryRetriever ready.")
    _data_loaded = True


def get_hotels_df():
    """Get the hotels dataframe, loading data if necessary."""
    _load_data()
    return df


def get_hotel_info_retriever():
    """Get the hotel info retriever, loading data if necessary."""
    _load_data()
    return hotel_info_retriever


def get_reviews_retriever():
    """Get the reviews retriever, loading data if necessary."""
    _load_data()
    return reviews_retriever



class SimpleInMemoryRetriever:
    """
    Minimal in-memory retriever:
    - embeds all docs once
    - on query: embeds query, computes dot-product similarities, returns top-k docs
    """

    def __init__(self, docs: List[Document], embeddings, k: int = 8):
        self.docs = docs
        self.embeddings = embeddings
        self.k = k

        if docs:
            texts = [d.page_content for d in docs]
            self.doc_vectors = np.array(self.embeddings.embed_documents(texts))
        else:
            self.doc_vectors = np.zeros((0, 0), dtype=float)

    def get_relevant_documents(self, query: str) -> List[Document]:
        if not self.docs:
            return []
        q_vec = np.array(self.embeddings.embed_query(query))
        scores = self.doc_vectors @ q_vec
        k = min(self.k, len(self.docs))
        idxs = np.argsort(scores)[::-1][:k]
        return [self.docs[i] for i in idxs]



session_store: Dict[str, InMemoryChatMessageHistory] = {}

def get_history(thread_id: str) -> InMemoryChatMessageHistory:
    """Get or create chat history for a thread."""
    if thread_id not in session_store:
        session_store[thread_id] = InMemoryChatMessageHistory()
    return session_store[thread_id]

def get_limited_history_text(
    history_obj: InMemoryChatMessageHistory,
    max_messages: int = 10,
) -> str:
    """Convert chat history to formatted text string."""
    msgs = history_obj.messages[-max_messages:]
    return "\n".join([f"{m.type.upper()}: {m.content}" for m in msgs])



def truncate_text(text: str, max_chars: int = 5000) -> str:
    """Truncate text to maximum character limit."""
    return text if len(text) <= max_chars else text[:max_chars] + " ...[TRUNCATED]"

def build_context_text(info_docs: Iterable[Document], review_docs: Iterable[Document]) -> str:
    """Build formatted context from hotel info and review documents."""
    # Convert to lists to avoid iterator exhaustion
    info_docs_list = list(info_docs)
    review_docs_list = list(review_docs)
    
    info_text = "\n\n".join(truncate_text(d.page_content, max_chars=5000) for d in info_docs_list)
    review_text = "\n\n".join(truncate_text(d.page_content, max_chars=5000) for d in review_docs_list)

    sections = []
    if info_text.strip():
        sections.append("HOTEL INFO:\n" + info_text)
    if review_text.strip():
        sections.append("GUEST REVIEWS:\n" + review_text)
    

    context = "\n\n".join(sections) if sections else "No relevant hotel context found."
    logger.info("Context built", length=len(context), info_docs=len(info_docs_list), review_docs=len(review_docs_list))
    
    return context



def detect_comparison_intent(text: str) -> bool:
    """Detect if user wants to compare hotels."""
    text = text.lower()
    comparison_keywords = [
        "compare", "comparison", "difference", "better", "vs", "versus",
        "which hotel", "best hotel", "alternatives", "option", "options",
    ]
    return any(kw in text for kw in comparison_keywords)


def detect_booking_intent(text: str) -> bool:
    """Detect booking / reservation intent."""
    t = text.lower()
    booking_keywords = [
        "book", "booking", "reserve", "reservation",
        "i want to stay", "i want to book", "i want this hotel",
        "i'll take this", "i will take this",
        "i want the first one", "i want the second one", "i want the third one",
    ]
    return any(kw in t for kw in booking_keywords)



async def resolve_query_with_context(user_message: str, state: Dict[str, Any], history_text: str) -> str:
    """
    Legacy/fallback function to resolve contextual references.
    Mainly used as backup if metadata agent is bypassed.
    
    Args:
        user_message: The user's query message
        state: HotelIQState containing last_suggestions
        history_text: Conversation history text
    
    Returns:
        Resolved query with hotel references replaced
    """
    text = user_message.lower()
    

    contextual_refs = [
        "this hotel", "that hotel", "the hotel", "this one", "that one",
        "it", "its", "the amenities", "their", "they", "there"
    ]
    has_reference = any(ref in text for ref in contextual_refs)
    
    if not has_reference:
        return user_message
    
    suggestions = state.get("last_suggestions", [])
    if not suggestions:
        return user_message
    
    most_recent_hotel = suggestions[-1]
    hotel_name = most_recent_hotel.get("name", "")
    
    if not hotel_name:
        return user_message
    
    prompts = get_prompts()
    rewrite_prompt = prompts.format(
        "utils.resolve_query_with_context",
        history_text=history_text,
        hotel_name=hotel_name,
        user_message=user_message
    )
    
    try:
        response = await llm.ainvoke(rewrite_prompt)
        rewritten = response.content.strip()
        if len(rewritten) > len(user_message) * 3 or hotel_name.lower() not in rewritten.lower():
            # Fallback
            rewritten = user_message.replace("this hotel", hotel_name)
            rewritten = rewritten.replace("that hotel", hotel_name)
        return rewritten
    except Exception:
        rewritten = user_message.replace("this hotel", hotel_name)
        return rewritten



def pick_hotel_for_booking(user_message: str, state: Dict[str, Any]) -> Optional[Dict[str, str]]:
    """
    Try to figure out which hotel the user wants to book:
    1) Match hotel name from text
    2) If they say 'first one' / 'second one', use state["last_suggestions"]
    
    Args:
        user_message: The user's booking request message
        state: HotelIQState containing last_suggestions
    
    Returns:
        Hotel dict with hotel_id, name, and star_rating, or None if not found
    """
    text = user_message.lower()

    # Get the hotels dataframe
    hotels_data = get_hotels_df()

    best_match = None
    for _, row in hotels_data.iterrows():
        name = str(row["official_name"])
        if not name.strip():
            continue
        if name.lower() in text:
            best_match = {
                "hotel_id": str(row["hotel_id"]),
                "name": name,
                "star_rating": str(row["star_rating"]),
            }
            break
    if best_match:
        return best_match


    suggestions = state.get("last_suggestions", [])
    if not suggestions:
        return None

    contextual_refs = ["this hotel", "that hotel", "this one", "that one", "this", "that"]
    has_contextual_ref = any(ref in text for ref in contextual_refs)
    
    if has_contextual_ref:
        return suggestions[-1] if suggestions else None
    

    idx = None
    if "first" in text or "1st" in text:
        idx = 0
    elif "second" in text or "2nd" in text:
        idx = 1
    elif "third" in text or "3rd" in text:
        idx = 2

    if idx is not None and idx < len(suggestions):
        return suggestions[idx]

    return suggestions[-1] if suggestions else None



def _get_comparison_prompt():
    """Load comparison prompt from YAML file."""
    prompts = get_prompts()
    prompt_text = prompts.get("comparison_agent.main_prompt")
    return ChatPromptTemplate.from_template(prompt_text)


comparison_prompt = _get_comparison_prompt()
comparison_chain = comparison_prompt | llm | StrOutputParser()

