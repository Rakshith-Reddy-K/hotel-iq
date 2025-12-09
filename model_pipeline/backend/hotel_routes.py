"""
Hotel Listing Routes for HotelIQ Frontend
Provides simplified hotel data endpoints
"""
from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel
from typing import List
from sql.hotel_queries import get_all_hotels_simplified, get_hotel_by_id_simplified

router = APIRouter(prefix="/hotels", tags=["Hotel Listings"])

class HotelSimplified(BaseModel):
    """Simplified hotel response model for frontend."""
    id: int
    name: str
    location: str
    rating: float
    reviews: int
    price: int
    image: str
    description: str
    amenities: List[str]


@router.get("/", response_model=List[HotelSimplified])
async def list_hotels():
    """
    Get all hotels with simplified data for frontend listing.
    
    Returns:
        List of hotels with basic information
    """
    try:
        hotels = get_all_hotels_simplified()
        return hotels
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to fetch hotels: {str(e)}"
        )


@router.get("/{hotel_id}", response_model=HotelSimplified)
async def get_hotel_details(hotel_id: int):
    """
    Get a specific hotel by ID with simplified data.
    
    Args:
        hotel_id: Hotel identifier
        
    Returns:
        Hotel details
    """
    try:
        hotel = get_hotel_by_id_simplified(hotel_id)
        
        if not hotel:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Hotel with id {hotel_id} not found"
            )
        
        return hotel
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to fetch hotel: {str(e)}"
        )


# Export router
hotel_router = router