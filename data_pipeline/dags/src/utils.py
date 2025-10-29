import json
import os
import re
import logging
from typing import Dict, List, Optional, Tuple

import pandas as pd
import requests
from dotenv import load_dotenv

# Configure logging
logger = logging.getLogger(__name__)

load_dotenv()


# ======================= HOTEL OFFERING UTILITIES =======================

def load_offering_json(file_path: str) -> pd.DataFrame:
    hotel_data: List[Dict] = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            try:
                hotel_data.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"Warning: Skipping invalid JSON line: {e}")
                continue

    df = pd.DataFrame(hotel_data)

    if 'address' in df.columns and df['address'].notna().any():
        address_df = df['address'].apply(pd.Series)
        address_df.columns = [f'address_{col}' for col in address_df.columns]
        df = pd.concat([df.drop('address', axis=1), address_df], axis=1)

    return df


def get_sample_hotels_by_city(
    df: pd.DataFrame,
    city_name: str,
    sample_size: int = 25,
    random_seed: int = 42
) -> pd.DataFrame:
    if 'address_locality' in df.columns:
        city_df = df[df['address_locality'].str.contains(city_name, case=False, na=False)]
    elif 'locality' in df.columns:
        city_df = df[df['locality'].str.contains(city_name, case=False, na=False)]
    else:
        raise ValueError("Could not find city column in DataFrame")

    if len(city_df) < sample_size:
        return city_df

    return city_df.sample(n=sample_size, random_state=random_seed)


# ======================= REVIEWS UTILITIES =======================

def clean_review_text(text):
    if not isinstance(text, str):
        return text
    text = re.sub(r'""+', '"', text)
    text = text.replace('"', '""')
    return text


def clean_reviews_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df_cleaned = df.copy()
    text_columns = ['title', 'review_text', 'reviewer_name']
    for col in text_columns:
        if col in df_cleaned.columns:
            df_cleaned[col] = df_cleaned[col].apply(clean_review_text)
    return df_cleaned


def load_reviews_json(file_path: str) -> pd.DataFrame:
    review_data: List[Dict] = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            try:
                review_data.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"Warning: Skipping invalid JSON line: {e}")
                continue

    df = pd.DataFrame(review_data)

    if 'ratings' in df.columns and df['ratings'].notna().any():
        ratings_df = df['ratings'].apply(pd.Series)
        if 'overall' in ratings_df.columns:
            df['rating_overall'] = ratings_df['overall']
        if 'service' in ratings_df.columns:
            df['rating_service'] = ratings_df['service']
        if 'cleanliness' in ratings_df.columns:
            df['rating_cleanliness'] = ratings_df['cleanliness']
        if 'value' in ratings_df.columns:
            df['rating_value'] = ratings_df['value']
        if 'location' in ratings_df.columns:
            df['rating_location'] = ratings_df['location']
        df = df.drop('ratings', axis=1)

    if 'author' in df.columns and df['author'].notna().any():
        author_df = df['author'].apply(pd.Series)
        if 'username' in author_df.columns:
            df['reviewer_name'] = author_df['username']
        df = df.drop('author', axis=1)

    df = df[[
        'id', 'offering_id', 'text', 'reviewer_name', 'date',
        'rating_overall', 'rating_service', 'rating_cleanliness', 'rating_value', 'rating_location'
    ]]

    df.columns = [
        'review_id', 'hotel_id', 'review_text', 'reviewer_name', 'review_date',
        'overall_rating', 'service_rating', 'cleanliness_rating', 'value_rating', 'location_rating'
    ]

    df['source'] = 'default'

    df = df[[
        'review_id', 'hotel_id', 'overall_rating', 'review_text',
        'reviewer_name', 'review_date', 'source',
        'service_rating', 'cleanliness_rating', 'value_rating', 'location_rating'
    ]]

    return df


def calculate_hotel_ratings(hotel_id: int, reviews_df: Optional[pd.DataFrame] = None, csv_path: str = 'reviews.csv') -> Dict[str, Optional[float]]:
    if reviews_df is None:
        reviews_df = pd.read_csv(csv_path)

    hotel_reviews = reviews_df[reviews_df['hotel_id'] == hotel_id]
    if len(hotel_reviews) == 0:
        return {
            'overall_rating': None,
            'total_reviews': 0,
            'cleanliness_rating': None,
            'service_rating': None,
            'location_rating': None,
            'value_rating': None
        }

    ratings = {
        'overall_rating': round(hotel_reviews['overall_rating'].mean(), 2) if pd.notna(hotel_reviews['overall_rating'].mean()) else None,
        'total_reviews': len(hotel_reviews),
        'cleanliness_rating': round(hotel_reviews['cleanliness_rating'].mean(), 2) if pd.notna(hotel_reviews['cleanliness_rating'].mean()) else None,
        'service_rating': round(hotel_reviews['service_rating'].mean(), 2) if pd.notna(hotel_reviews['service_rating'].mean()) else None,
        'location_rating': round(hotel_reviews['location_rating'].mean(), 2) if pd.notna(hotel_reviews['location_rating'].mean()) else None,
        'value_rating': round(hotel_reviews['value_rating'].mean(), 2) if pd.notna(hotel_reviews['value_rating'].mean()) else None
    }
    return ratings


def calculate_all_hotel_ratings(reviews_df: Optional[pd.DataFrame] = None, csv_path: str = 'reviews.csv') -> pd.DataFrame:
    if reviews_df is None:
        reviews_df = pd.read_csv(csv_path)

    hotel_ratings = reviews_df.groupby('hotel_id').agg({
        'overall_rating': 'mean',
        'service_rating': 'mean',
        'cleanliness_rating': 'mean',
        'value_rating': 'mean',
        'location_rating': 'mean',
        'review_id': 'count'
    }).reset_index()

    hotel_ratings.columns = ['hotel_id', 'overall_rating', 'service_rating', 'cleanliness_rating', 'value_rating', 'location_rating', 'total_reviews']
    rating_cols = ['overall_rating', 'service_rating', 'cleanliness_rating', 'value_rating', 'location_rating']
    hotel_ratings[rating_cols] = hotel_ratings[rating_cols].round(2)
    return hotel_ratings


def get_reviews_for_hotels(hotel_csv_path: str, reviews_csv_path: str, output_csv_path: str) -> pd.DataFrame:
    hotels_df = pd.read_csv(hotel_csv_path)
    hotel_ids = hotels_df['id'].tolist()
    reviews_df = pd.read_csv(reviews_csv_path)
    filtered_reviews = reviews_df[reviews_df['hotel_id'].isin(hotel_ids)]
    filtered_reviews['review_text'] = filtered_reviews[['review_text']].applymap(clean_review_text)
    filtered_reviews.to_csv(output_csv_path, index=False)
    return filtered_reviews


# ======================= ENRICHMENT UTILITIES =======================

def validate_and_fix_json(json_string: str) -> Optional[Dict]:
    """
    Validate and attempt to fix common JSON issues
    
    Args:
        json_string: Raw JSON string to validate and fix
        
    Returns:
        Parsed JSON dict if successful, None if failed
    """
    try:
        # First try to parse as-is
        return json.loads(json_string)
    except json.JSONDecodeError as e:
        logger.warning(f"Initial JSON parse failed: {e}")
        
        # Try common fixes
        fixed_json = json_string
        
        # Fix 1: Remove trailing commas
        fixed_json = re.sub(r',\s*}', '}', fixed_json)
        fixed_json = re.sub(r',\s*]', ']', fixed_json)
        
        # Fix 2: Fix unquoted property names
        fixed_json = re.sub(r'(\w+):', r'"\1":', fixed_json)
        
        # Fix 3: Fix single quotes to double quotes
        fixed_json = re.sub(r"'([^']*)':", r'"\1":', fixed_json)
        
        # Fix 4: Remove any content after the last }
        last_brace = fixed_json.rfind('}')
        if last_brace != -1:
            fixed_json = fixed_json[:last_brace + 1]
        
        try:
            return json.loads(fixed_json)
        except json.JSONDecodeError as e2:
            logger.error(f"JSON still invalid after fixes: {e2}")
            logger.error(f"Original content: {json_string[:500]}")
            logger.error(f"Fixed content: {fixed_json[:500]}")
            return None

def extract_hotel_data(hotel_name: str, location: str, api_key: Optional[str] = None) -> Optional[Dict]:
    """
    Extract comprehensive hotel data using Perplexity API.
    
    Args:
        hotel_name (str): Name of the hotel
        location (str): Location of the hotel (e.g., "Los Angeles")
        api_key (str, optional): Perplexity API key. If not provided, will look for PERPLEXITY_API_KEY env variable.
        
    Returns:
        dict: JSON response containing detailed hotel information
    """
    logger.info(f"Starting Perplexity API call for hotel: {hotel_name} in {location}")
    
    if api_key is None:
        api_key = os.getenv('PERPLEXITY_API_KEY')
        if api_key is None:
            logger.error("Perplexity API key not provided. Set PERPLEXITY_API_KEY environment variable or pass as parameter.")
            raise ValueError("Perplexity API key not provided. Set PERPLEXITY_API_KEY environment variable or pass as parameter.")

    url = "https://api.perplexity.ai/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    prompt = f"""You are a hotel data researcher. I need you to gather comprehensive information about the following hotel:

**Hotel Name:** {hotel_name}
**Location:** {location}

Please search the web and compile the following information in a structured JSON format that matches a specific database schema. Prioritize official sources (hotel website, Expedia, Booking.com, TripAdvisor) over user-generated content.

**REQUIRED INFORMATION:**

1. **Hotel Main Information:**
   - Official full name (max 255 characters)
   - Star rating (integer between 1-5)
   - Description: Write 2-3 cohesive paragraphs about the hotel's essence, design, atmosphere, and history. This should read like premium travel writing with NO bullet points or lists.
   - Full street address (complete address as single text)
   - Year opened (DATE format: YYYY-MM-DD, use January 1st if only year known)
   - Last renovation date (DATE format: YYYY-MM-DD, use January 1st if only year known)
   - Total number of rooms/suites (integer)
   - Number of floors (integer)
   - Additional Information: Write 2-3 cohesive paragraphs covering neighborhood context, target guests, dining details, and brand positioning. Write as flowing prose with NO bullet points or lists.

2. **Room Types:**
   For each room type, provide:
   - Room type name (max 100 characters)
   - Bed configuration (max 100 characters, e.g., "1 King", "2 Queens")
   - Room size in square feet (integer only, no units)
   - Maximum occupancy (integer)
   - Minimum price (decimal, e.g., 189.00)
   - Maximum price (decimal, e.g., 279.00)
   - Room description (text, can include special features)

3. **Amenities:**
   Organize amenities by category. For each amenity:
   - Category (max 50 characters, e.g., "Room Amenities", "Property Amenities", "Fitness", "Business", "Dining")
   - Description (text summary of amenities in this category)
   - Details (JSON object with specific key-value pairs for this category)

   **Categories to include:**
   - Room Amenities (WiFi, TV, coffee maker, mini-fridge, microwave, safe, iron, hair dryer, toiletries, AC, heating, workspace)
   - Property Amenities (pool, fitness center, spa, business center, meeting space, parking, EV charging, accessibility)
   - Dining (restaurants and bars with their details)
   - Services (front desk, shuttle, concierge, laundry, etc.)

4. **Policies:**
   - Check-in time (TIME format: HH:MM:SS, e.g., "15:00:00")
   - Check-out time (TIME format: HH:MM:SS, e.g., "11:00:00")
   - Minimum age requirement (integer)
   - Pet policy (text description)
   - Smoking policy (text description)
   - Children policy (text description)
   - Extra person policy (text description including fees)
   - Cancellation policy (text description)

5. **Special Features & Awards:**
   - List of special features/programs
   - List of awards and certifications

**FORMAT REQUIREMENTS:**

Return the information as a valid JSON object following this EXACT structure:
```json
{{
  "hotel": {{
    "official_name": "Full official hotel name",
    "star_rating": 4,
    "description": "2-3 flowing paragraphs about hotel essence, design, atmosphere, and history. NO bullet points.",
    "address": "Complete street address",
    "year_opened": "2015-01-01",
    "last_renovation": "2021-01-01",
    "total_rooms": 261,
    "number_of_floors": 5,
    "additional_info": "2-3 flowing paragraphs covering neighborhood, target guests, dining venues, and brand positioning. NO bullet points."
  }},
  "rooms": [
    {{
      "room_type": "Room, 1 King Bed",
      "bed_configuration": "1 King",
      "room_size_sqft": 280,
      "max_occupancy": 2,
      "price_range_min": 189.00,
      "price_range_max": 279.00,
      "description": "Renovated room with blackout curtains, sofa bed, refrigerator, connecting rooms available"
    }}
  ],
  "amenities": [
    {{
      "category": "Room Amenities",
      "description": "All rooms include modern conveniences for comfort and productivity",
      "details": {{
        "wifi": "Free high-speed",
        "tv": "43-inch flat-screen with premium channels",
        "coffee_maker": "Keurig",
        "mini_fridge": true,
        "microwave": true,
        "safe": "At front desk",
        "iron_ironing_board": true,
        "hair_dryer": true,
        "toiletries_brand": "Standard Hilton",
        "air_conditioning": true,
        "heating": true,
        "workspace_desk": "Ergonomic desk with chair"
      }}
    }},
    {{
      "category": "Property Amenities",
      "description": "Hotel features for guests during their stay",
      "details": {{
        "pool": "Indoor heated pool",
        "fitness_center": "24-hour with cardio and weights",
        "spa": null,
        "business_center": "24-hour with computers and printer",
        "meeting_space": "3 rooms totaling 2099 sq ft",
        "parking": "Self parking $20/day with EV charging",
        "accessibility": "Mobility and hearing accessible rooms and public areas"
      }}
    }},
    {{
      "category": "Dining",
      "description": "On-site dining options",
      "details": {{
        "restaurants": [
          {{
            "name": "Garden Grille & Bar",
            "cuisine": "American",
            "meal_service": "Breakfast 6-10am Mon-Fri, 7-11am Sat-Sun; Dinner 5-10pm",
            "price_range": "$$",
            "special_features": "Casual dining with kid menu and bar service"
          }}
        ]
      }}
    }},
    {{
      "category": "Services",
      "description": "Guest services available",
      "details": {{
        "front_desk": "24-hour",
        "multilingual_staff": true,
        "airport_shuttle": "Free 24-hour shuttle",
        "baggage_storage": true,
        "concierge": true,
        "laundry": "Self-service and dry cleaning available",
        "express_checkin_checkout": true
      }}
    }}
  ],
  "policies": {{
    "check_in_time": "15:00:00",
    "check_out_time": "11:00:00",
    "min_age_requirement": 21,
    "pet_policy": "Dogs and cats allowed, maximum 2 pets, $75 fee per stay, service animals free",
    "smoking_policy": "Non-smoking property",
    "children_policy": "Children welcome, cribs and infant beds available free on request",
    "extra_person_policy": "$10 per stay for rollaway bed",
    "cancellation_policy": "Free cancellation up to 24 hours before arrival on standard rates; otherwise one night fee charged"
  }},
  "special_features": [
    "Hilton Honors rewards program",
    "EV charging stations",
    "Family, romance, and park & fly packages",
    "COVID-19 safety measures with enhanced cleaning",
    "Wedding and meeting/event services"
  ],
  "awards": [
    "TripAdvisor Travelers' Choice",
    "Consistently rated 9.2+ on Expedia"
  ]
}}
```

**CRITICAL DATA TYPE REQUIREMENTS:**

1. **Integers:** star_rating, room_size_sqft, max_occupancy, total_rooms, number_of_floors, min_age_requirement
2. **Decimals:** price_range_min, price_range_max (use format XX.XX)
3. **Dates:** year_opened, last_renovation (format: YYYY-MM-DD, if only year known use January 1st)
4. **Time:** check_in_time, check_out_time (format: HH:MM:SS in 24-hour format)
5. **Text:** description, additional_info, address, policy descriptions
6. **Varchar limits:** official_name (255), room_type (100), bed_configuration (100), category (50)
7. **Boolean in JSON:** Use true/false (lowercase)
8. **Null values:** Use null (lowercase) for missing data, not "not_available" or empty strings

**IMPORTANT GUIDELINES:**

1. **Description Field (about section):**
   - Must be 2-3 flowing paragraphs
   - Focus ONLY on: property essence, design/atmosphere, history/credentials
   - NO bullet points, NO lists
   - Write in engaging, professional tone

2. **Additional_info Field:**
   - Must be 2-3 flowing paragraphs
   - Cover: neighborhood/nearby attractions, target guests/experience, dining details, brand positioning
   - NO bullet points, NO lists
   - Write as connected prose

3. **Room Size:**
   - Must be integer only (no "sq ft" or "approx")
   - Extract numeric value only (e.g., "280 sq ft" becomes 280)

4. **Prices:**
   - Must be decimal format (189.00, not "$189" or "189-279")
   - Separate min and max clearly

5. **Dates:**
   - Always use YYYY-MM-DD format
   - If only year available, use January 1st (e.g., "2015" becomes "2015-01-01")

6. **Times:**
   - Always use HH:MM:SS format in 24-hour time
   - "3:00 PM" becomes "15:00:00"
   - "11:00 AM" becomes "11:00:00"

7. **Amenities Structure:**
   - Group related amenities into categories
   - Each category should have: category name, description, and details JSON object
   - Use consistent category names

8. **Source Prioritization:**
   - Hotel's official website
   - Major booking platforms (Expedia, Booking.com, Hotels.com)
   - Travel guides

9. **Data Validation:**
   - Verify data types match requirements
   - Use null for missing data, not empty strings or "Not Avaliable"
   - Ensure all varchar fields don't exceed character limits
   - Cross-reference information from multiple sources

Begin your search and provide the complete JSON response matching the exact schema structure above."""

    payload = {
        "model": "sonar-pro",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant that returns structured JSON data."},
            {"role": "user", "content": prompt}
        ],
      "search_domain_filter":[
        "expedia.com",
        "booking.com"
    ],
        "temperature": 0.2,
        "max_tokens": 8000,

    }

    try:
        logger.info(f"Making API request for hotel: {hotel_name}")
        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status()
        
        result = response.json()
        content = result['choices'][0]['message']['content']
        
        # Extract JSON from markdown code blocks
        if '```json' in content:
            json_start = content.find('```json') + 7
            json_end = content.find('```', json_start)
            if json_end == -1:
                logger.error(f"No closing ``` found for JSON block in hotel: {hotel_name}")
                return None
            content = content[json_start:json_end].strip()
        elif '```' in content:
            json_start = content.find('```') + 3
            json_end = content.find('```', json_start)
            if json_end == -1:
                logger.error(f"No closing ``` found for code block in hotel: {hotel_name}")
                return None
            content = content[json_start:json_end].strip()
        
        # Log the extracted content for debugging
        logger.debug(f"Extracted content for {hotel_name}: {content[:200]}...")
        
        # Validate JSON before parsing
        if not content.strip():
            logger.error(f"Empty content extracted for hotel: {hotel_name}")
            return None
            
        # Try to parse JSON with better error handling
        hotel_data = validate_and_fix_json(content)
        if hotel_data:
            logger.info(f"Successfully retrieved data for hotel: {hotel_name}")
            return hotel_data
        else:
            logger.error(f"Failed to parse JSON for hotel: {hotel_name}")
            return None
        
    except Exception as e:
        logger.error(f"Failed to retrieve data for hotel {hotel_name}: {str(e)}")
        return None


def extract_hotel_data_from_row(df_row: pd.Series) -> Dict:
    hotel_name = df_row.get('name', '')
    location = df_row.get('address_locality', '')
    if not hotel_name:
        raise ValueError("Hotel name not found in DataFrame row")
    if not location:
        raise ValueError("Location not found in DataFrame row")
    return extract_hotel_data(hotel_name, location)


def merge_hotel_data(df_row: pd.Series, hotel_data_json: Dict) -> Dict[str, pd.DataFrame]:
    hotel_info = hotel_data_json.get('hotel', {}) if hotel_data_json else {}

    hotels_data = {
        'hotel_id': df_row.get('id'),
        'official_name': hotel_info.get('official_name', df_row.get('name')),
        'star_rating': hotel_info.get('star_rating') or df_row.get('hotel_class'),
        'description': hotel_info.get('description'),
        'address': df_row.get('address_street-address'),
        'city': df_row.get('address_locality'),
        'state': df_row.get('address_region'),
        'zip_code': df_row.get('address_postal-code'),
        'country': 'USA',
        'phone': df_row.get('phone'),
        'email': None,
        'website': df_row.get('url'),
        'overall_rating': None,
        'total_reviews': 0,
        'cleanliness_rating': None,
        'service_rating': None,
        'location_rating': None,
        'value_rating': None,
        'year_opened': hotel_info.get('year_opened'),
        'last_renovation': hotel_info.get('last_renovation'),
        'total_rooms': hotel_info.get('total_rooms'),
        'number_of_floors': hotel_info.get('number_of_floors'),
        'additional_info': hotel_info.get('additional_info')
    }

    rooms_list = (hotel_data_json or {}).get('rooms', [])
    rooms_data: List[Dict] = []
    for room in rooms_list:
        rooms_data.append({
            'hotel_id': df_row.get('id'),
            'room_type': room.get('room_type'),
            'bed_configuration': room.get('bed_configuration'),
            'room_size_sqft': room.get('room_size_sqft'),
            'max_occupancy': room.get('max_occupancy'),
            'price_range_min': room.get('price_range_min'),
            'price_range_max': room.get('price_range_max'),
            'description': room.get('description')
        })

    amenities_list = (hotel_data_json or {}).get('amenities', [])
    amenities_data: List[Dict] = []
    for amenity in amenities_list:
        amenities_data.append({
            'hotel_id': df_row.get('id'),
            'category': amenity.get('category'),
            'description': amenity.get('description'),
            'details': json.dumps(amenity.get('details')) if amenity.get('details') else None
        })

    policies_info = (hotel_data_json or {}).get('policies', {})
    policies_df = pd.DataFrame([{
        'hotel_id': df_row.get('id'),
        'check_in_time': policies_info.get('check_in_time'),
        'check_out_time': policies_info.get('check_out_time'),
        'min_age_requirement': policies_info.get('min_age_requirement'),
        'pet_policy': policies_info.get('pet_policy'),
        'smoking_policy': policies_info.get('smoking_policy'),
        'children_policy': policies_info.get('children_policy'),
        'extra_person_policy': policies_info.get('extra_person_policy'),
        'cancellation_policy': policies_info.get('cancellation_policy')
    }]) if any(policies_info.values()) else pd.DataFrame()

    return {
        'hotels': pd.DataFrame([hotels_data]),
        'rooms': pd.DataFrame(rooms_data) if rooms_data else pd.DataFrame(),
        'amenities': pd.DataFrame(amenities_data) if amenities_data else pd.DataFrame(),
        'policies': policies_df,
    }


def export_hotel_data_to_csv(merged_data: Dict[str, pd.DataFrame], output_dir: str = 'output') -> None:
    os.makedirs(output_dir, exist_ok=True)
    for table in ['hotels', 'rooms', 'amenities', 'policies']:
        df = merged_data.get(table)
        if df is not None and not df.empty:
            df.to_csv(os.path.join(output_dir, f'{table}.csv'), index=False)

