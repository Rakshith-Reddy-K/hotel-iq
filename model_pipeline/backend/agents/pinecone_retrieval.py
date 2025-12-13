"""
Pinecone Retrieval Functions
=============================

Handles retrieval from Pinecone vector databases for hotels and reviews.
"""

import os
from typing import List, Dict, Any, Optional
from pinecone import Pinecone
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from dotenv import load_dotenv
load_dotenv()
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from logger_config import get_logger

logger = get_logger(__name__)


def get_pinecone_client():
    """Initialize and return Pinecone client."""
    api_key = os.getenv("PINECONE_API_KEY")
    if not api_key:
        raise ValueError("PINECONE_API_KEY environment variable not set")
    
    pc = Pinecone(api_key=api_key)
    return pc


embeddings = OpenAIEmbeddings(
    model="text-embedding-3-large",  # 3072 dimensions
    openai_api_key=os.getenv("OPENAI_API_KEY")
)


def retrieve_hotels_by_description(
    query: str,
    top_k: int = 5,
    filter_dict: Optional[Dict[str, Any]] = None
) -> List[Document]:
    """
    Retrieve hotels based on description similarity.
    
    Args:
        query: Search query (e.g., "luxury hotel with pool near downtown")
        top_k: Number of results to return (default: 5)
        filter_dict: Optional metadata filters (e.g., {"city": "Boston"})
        
    Returns:
        List of Document objects with hotel information
        
    Example:
        >>> hotels = retrieve_hotels_by_description("hotel with ocean view", top_k=3)
        >>> for hotel in hotels:
        ...     print(hotel.metadata["hotel_name"], hotel.metadata["address"])
    """
    try:
        pc = get_pinecone_client()
        index_name = os.getenv("HOTEL_INDEX_NAME")
        index = pc.Index(index_name)
        
        query_embedding = embeddings.embed_query(query)
        
        results = index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True,
            filter=filter_dict
        )
        
        documents = []
        for match in results.matches:
            metadata = match.metadata
            

            
            description = metadata.get("description", "")
            if not description:
                # Reconstruct description from available fields
                hotel_name = metadata.get("hotel_name") or metadata.get("official_name") or metadata.get("name") or "Unknown Hotel"
                description = f"{hotel_name} located at {metadata.get('address', 'Unknown Address')}"
            
            doc = Document(
                page_content=description,
                metadata={
                    "hotel_id": metadata.get("hotel_id", ""),
                    "hotel_name": metadata.get("hotel_name") or metadata.get("official_name") or metadata.get("name", ""),
                    "name": metadata.get("hotel_name") or metadata.get("official_name") or metadata.get("name", ""),
                    "official_name": metadata.get("official_name") or metadata.get("hotel_name") or metadata.get("name", ""),
                    "star_rating": metadata.get("star_rating", "N/A"),
                    "city": metadata.get("city", ""),
                    "state": metadata.get("state", ""),
                    "zip_code": metadata.get("zip_code", ""),
                    "address": metadata.get("address", ""),
                    "phone": metadata.get("phone", ""),
                    "website": metadata.get("website", ""),
                    "overall_rating": metadata.get("overall_rating", "N/A"),
                    "total_reviews": metadata.get("total_reviews", ""),
                    "score": match.score  # Similarity score
                }
            )
            documents.append(doc)
        
        logger.info("Retrieved hotels from Pinecone", count=len(documents))
        return documents
        
    except Exception as e:
        logger.error("Error retrieving hotels from Pinecone", error=str(e))
        return []


def retrieve_reviews_by_query(
    query: str,
    hotel_id: Optional[str] = None,
    top_k: int = 10,
    filter_dict: Optional[Dict[str, Any]] = None
) -> List[Document]:
    """
    Retrieve reviews based on query similarity.
    
    Args:
        query: Search query (e.g., "reviews about cleanliness")
        hotel_id: Optional hotel_id to filter reviews for specific hotel
        top_k: Number of results to return (default: 10)
        filter_dict: Optional metadata filters (e.g., {"overall_rating": {"$gte": 4}})
        
    Returns:
        List of Document objects with review information
        
    Example:
        >>> reviews = retrieve_reviews_by_query("great breakfast", hotel_id="123", top_k=5)
        >>> for review in reviews:
        ...     print(f"Rating: {review.metadata['overall_rating']}")
        ...     print(review.page_content[:100])
    """
    try:
        # Get Pinecone client
        pc = get_pinecone_client()
        index_name = os.getenv("REVIEWS_INDEX_NAME")
        index = pc.Index(index_name)
        
        # Generate query embedding
        query_embedding = embeddings.embed_query(query)
        
        final_filter = filter_dict or {}
        if hotel_id:
            if final_filter:
                final_filter = {"$and": [final_filter, {"hotel_id": hotel_id}]}
            else:
                final_filter = {"hotel_id": hotel_id}
        
        # Query Pinecone
        results = index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True,
            filter=final_filter if final_filter else None
        )
        
        # Convert to LangChain Document objects
        documents = []
        for match in results.matches:
            # Extract metadata
            metadata = match.metadata
            
            review_text = metadata.get("review_text", metadata.get("text", ""))
            
            doc = Document(
                page_content=review_text,
                metadata={
                    "hotel_id": metadata.get("hotel_id", ""),
                    "overall_rating": metadata.get("overall_rating", ""),
                    "rating": metadata.get("overall_rating", ""),  # Alias for compatibility
                    "review_date": metadata.get("review_date", ""),
                    "score": match.score  # Similarity score
                }
            )
            documents.append(doc)
        
        logger.info("Retrieved reviews from Pinecone", count=len(documents), hotel_id=hotel_id)
        return documents
        
    except Exception as e:
        logger.error("Error retrieving reviews from Pinecone", error=str(e))
        return []


def get_hotel_by_id(hotel_id: str) -> Optional[Document]:
    """
    Retrieve a specific hotel by its ID.
    
    Args:
        hotel_id: The hotel ID to retrieve
        
    Returns:
        Document object with hotel information, or None if not found
        
    Example:
        >>> hotel = get_hotel_by_id("123")
        >>> if hotel:
        ...     print(hotel.metadata["hotel_name"])
    """
    try:
        hotels = retrieve_hotels_by_description(
            query="hotel",  # Generic query since we're filtering by ID
            top_k=1,
            filter_dict={"hotel_id": hotel_id}
        )
        
        if hotels:
            return hotels[0]
        else:
            logger.warning("Hotel not found in Pinecone", hotel_id=hotel_id)
            return None
            
    except Exception as e:
        logger.error("Error retrieving hotel by ID", error=str(e))
        return None


def get_reviews_by_hotel_id(
    hotel_id: str,
    top_k: int = 20,
    min_rating: Optional[float] = None,
    max_rating: Optional[float] = None
) -> List[Document]:
    """
    Retrieve all reviews for a specific hotel.
    
    Args:
        hotel_id: The hotel ID to get reviews for
        top_k: Maximum number of reviews to return (default: 20)
        min_rating: Optional minimum rating filter (e.g., 4.0)
        max_rating: Optional maximum rating filter (e.g., 5.0)
        
    Returns:
        List of Document objects with review information
        
    Example:
        >>> reviews = get_reviews_by_hotel_id("123", min_rating=4.0)
        >>> print(f"Found {len(reviews)} reviews with rating >= 4.0")
    """
    try:
        filter_dict = None
        if min_rating is not None or max_rating is not None:
            rating_filter = {}
            if min_rating is not None:
                rating_filter["$gte"] = min_rating
            if max_rating is not None:
                rating_filter["$lte"] = max_rating
            filter_dict = {"overall_rating": rating_filter}
        
        reviews = retrieve_reviews_by_query(
            query="review",  # Generic query
            hotel_id=hotel_id,
            top_k=top_k,
            filter_dict=filter_dict
        )
        
        return reviews
        
    except Exception as e:
        logger.error("Error retrieving reviews for hotel", hotel_id=hotel_id, error=str(e))
        return []


def find_similar_hotels(
    hotel_id: str,
    top_k: int = 3,
    exclude_current: bool = True
) -> List[Document]:
    """
    Find hotels similar to a given hotel based on description.
    
    Args:
        hotel_id: The hotel ID to find similar hotels for
        top_k: Number of similar hotels to return
        exclude_current: Whether to exclude the current hotel from results
        
    Returns:
        List of Document objects with similar hotel information
        
    Example:
        >>> similar = find_similar_hotels("123", top_k=3)
        >>> for hotel in similar:
        ...     print(f"Similar hotel: {hotel.metadata['hotel_name']}")
    """
    try:
        current_hotel = get_hotel_by_id(hotel_id)
        
        if not current_hotel:
            logger.warning("Could not find hotel to find similar hotels", hotel_id=hotel_id)
            return []
        
        description = current_hotel.page_content
        
        search_k = top_k + 1 if exclude_current else top_k
        
        similar_hotels = retrieve_hotels_by_description(
            query=description,
            top_k=search_k
        )
        
        if exclude_current:
            similar_hotels = [
                hotel for hotel in similar_hotels 
                if hotel.metadata.get("hotel_id") != hotel_id
            ][:top_k]
        
        logger.info("Found similar hotels", count=len(similar_hotels), hotel_id=hotel_id)
        return similar_hotels
        
    except Exception as e:
        logger.error("Error finding similar hotels", error=str(e))
        return []


def test_hotel_retrieval():
    """Test hotel retrieval function."""
    print("\n=== Testing Hotel Retrieval ===")
    hotels = retrieve_hotels_by_description("luxury hotel with pool", top_k=3)
    for i, hotel in enumerate(hotels, 1):
        print(f"\n{i}. {hotel.metadata.get('hotel_name', 'Unknown')}")
        print(f"   Address: {hotel.metadata.get('address', 'N/A')}")
        print(f"   City: {hotel.metadata.get('city', 'N/A')}")
        print(f"   Score: {hotel.metadata.get('score', 0):.4f}")


def test_review_retrieval():
    """Test review retrieval function."""
    print("\n=== Testing Review Retrieval ===")
    reviews = retrieve_reviews_by_query("great service and clean rooms", top_k=3)
    for i, review in enumerate(reviews, 1):
        print(f"\n{i}. Rating: {review.metadata.get('overall_rating', 'N/A')}")
        print(f"   Date: {review.metadata.get('review_date', 'N/A')}")
        print(f"   Review: {review.page_content[:100]}...")
        print(f"   Score: {review.metadata.get('score', 0):.4f}")


if __name__ == "__main__":
    """Run tests when script is executed directly."""
    logger.info("Running Pinecone Retrieval Tests...")
    
    try:
        test_hotel_retrieval()
        test_review_retrieval()
        logger.info("Tests completed!")
    except Exception as e:
        logger.error("Test failed", error=str(e))

