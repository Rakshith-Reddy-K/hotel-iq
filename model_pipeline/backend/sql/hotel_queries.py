"""
Hotel Listing Queries for HotelIQ
Provides simplified hotel data for frontend display
"""
import json
from typing import List, Dict, Optional
from sql.db_pool import get_connection

def get_all_hotels_simplified() -> List[Dict]:
    """
    Fetch all hotels with simplified data for frontend listing.
    Returns: List of hotel dictionaries matching frontend format
    """
    with get_connection() as conn:
        cur = conn.cursor()
        
        cur.execute("""
            SELECT 
                hotel_id,
                official_name,
                address,
                city,
                overall_rating,
                total_reviews,
                description,
                images
            FROM hotels
            ORDER BY overall_rating DESC NULLS LAST
        """)
        
        hotels = cur.fetchall()
        result = []
        
        for hotel in hotels:
            hotel_id = hotel[0]
            
            # Get minimum price
            price = get_hotel_min_price(hotel_id, cur)
            
            # Get amenities
            amenities = extract_hotel_amenities(hotel_id, cur)
            
            # Get image
            image = extract_hotel_image(hotel[7])
            
            # Format location
            location = format_hotel_location(hotel[2], hotel[3])
            
            hotel_obj = {
                "id": hotel_id,
                "name": hotel[1],
                "location": location,
                "rating": round(float(hotel[4]), 1) if hotel[4] else 4.0,
                "reviews": hotel[5] if hotel[5] else 0,
                "price": price,
                "image": image,
                "description": truncate_text(hotel[6], 300),
                "amenities": amenities
            }
            
            result.append(hotel_obj)
        
        cur.close()
        return result


def get_hotel_by_id_simplified(hotel_id: int) -> Optional[Dict]:
    """
    Fetch a specific hotel by ID with simplified data.
    """
    with get_connection() as conn:
        cur = conn.cursor()
        
        cur.execute("""
            SELECT 
                hotel_id,
                official_name,
                address,
                city,
                overall_rating,
                total_reviews,
                description,
                images
            FROM hotels
            WHERE hotel_id = %s
        """, (hotel_id,))
        
        hotel = cur.fetchone()
        
        if not hotel:
            cur.close()
            return None
        
        price = get_hotel_min_price(hotel_id, cur)
        amenities = extract_hotel_amenities(hotel_id, cur)
        image = extract_hotel_image(hotel[7])
        location = format_hotel_location(hotel[2], hotel[3])
        
        hotel_obj = {
            "id": hotel_id,
            "name": hotel[1],
            "location": location,
            "rating": round(float(hotel[4]), 1) if hotel[4] else 4.0,
            "reviews": hotel[5] if hotel[5] else 0,
            "price": price,
            "image": image,
            "description": hotel[6] if hotel[6] else "",
            "amenities": amenities
        }
        
        cur.close()
        return hotel_obj


def get_hotel_min_price(hotel_id: int, cursor) -> int:
    """Get minimum price from rooms table."""
    cursor.execute("""
        SELECT MIN(price_range_min)
        FROM rooms
        WHERE hotel_id = %s AND price_range_min IS NOT NULL
    """, (hotel_id,))
    
    result = cursor.fetchone()
    return int(result[0]) if result and result[0] else 0


def extract_hotel_amenities(hotel_id: int, cursor) -> List[str]:
    """
    Extract amenities from amenities table.
    Returns list of 5-6 key amenities.
    """
    cursor.execute("""
        SELECT category, details
        FROM amenities
        WHERE hotel_id = %s
    """, (hotel_id,))
    
    amenities_data = cursor.fetchall()
    amenity_list = []
    
    for row in amenities_data:
        category = row[0]
        details = row[1]
        
        # Parse JSON details
        if isinstance(details, str):
            try:
                details = json.loads(details)
            except:
                continue
        
        if not isinstance(details, dict):
            continue
        
        # Extract based on category
        if category == "Room Amenities":
            if any(key in str(details).lower() for key in ["wifi", "wireless internet"]):
                amenity_list.append("Free Wifi")
            if "air_conditioning" in details or "air conditioning" in str(details).lower():
                amenity_list.append("Air Conditioning")
            if "refrigerator" in str(details).lower():
                amenity_list.append("Refrigerator")
                
        elif category == "Property Amenities":
            if any(key in str(details).lower() for key in ["fitness", "gym"]):
                amenity_list.append("Fitness Center")
            if "parking" in str(details).lower():
                amenity_list.append("Parking")
            if "pool" in str(details).lower():
                amenity_list.append("Pool")
                
        elif category == "Dining":
            if "restaurant" in str(details).lower():
                amenity_list.append("Restaurant")
            if "room service" in str(details).lower():
                amenity_list.append("Room Service")
                
        elif category == "Services":
            if "concierge" in str(details).lower():
                amenity_list.append("Concierge")
    
    # Remove duplicates, limit to 6
    amenity_list = list(dict.fromkeys(amenity_list))[:6]
    
    if not amenity_list:
        amenity_list = ["Free Wifi", "24/7 Front Desk", "Air Conditioning"]
    
    return amenity_list


def extract_hotel_image(images_string: Optional[str]) -> str:
    """Extract first image URL or return placeholder."""
    if images_string and images_string.strip():
        if ',' in images_string:
            images = images_string.split(',')
            return images[0].strip()
        return images_string.strip()
    
    return "https://images.unsplash.com/photo-1542314831-068cd1dbfeeb?ixlib=rb-4.0.3&auto=format&fit=crop&w=1000&q=80"


def format_hotel_location(address: Optional[str], city: Optional[str]) -> str:
    """Format location string."""
    parts = []
    
    if address:
        address_parts = address.split(',')
        if address_parts:
            parts.append(address_parts[0].strip())
    
    if city:
        parts.append(city)
    
    return ", ".join(parts) if parts else "Boston, MA"


def truncate_text(text: Optional[str], max_length: int = 300) -> str:
    """Truncate text to max length."""
    if not text:
        return ""
    
    if len(text) > max_length:
        return text[:max_length-3] + "..."
    
    return text