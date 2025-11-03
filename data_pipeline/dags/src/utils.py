import json
import os
import re
import logging
from typing import Dict, List, Optional, Tuple
from src.bucket_util import download_file_from_gcs
import pandas as pd
import requests
from dotenv import load_dotenv
import json
from datetime import datetime

# Configure logging
logger = logging.getLogger(__name__)

load_dotenv()


def check_if_filtering_needed():
    city = 'boston'
    try:
        download_file_from_gcs(f"filtered/{city}/hotels.csv", 
                              f"data/filtered/{city}/hotels.csv")
        download_file_from_gcs(f"filtered/{city}/reviews.csv", 
                              f"data/filtered/{city}/reviews.csv")
        
        # TODO: Check if raw data changed (using DVC or checksums)
        # For now, assume if filtered exists, we can skip
        logger.info("Filtered data exists - skipping filtering!")
        return 'skip_filtering'
        
    except:
        logger.info("Filtered data not found - need to filter!")
        return 'do_filtering'


def parse_raw_hotels(file_path: str) -> pd.DataFrame:
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

def parse_raw_reviews(file_path: str) -> pd.DataFrame:
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

def filter_hotels_by_city(
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

def clean_review_text(text):
    if not isinstance(text, str):
        return text
    text = re.sub(r'""+', '"', text)
    text = text.replace('"', '""')
    return text

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


def validate_and_fix_json(json_string: str) -> Optional[Dict]:
    """
    Validate and attempt to fix common JSON issues
    
    Args:
        json_string: Raw JSON string to validate and fix
        
    Returns:
        Parsed JSON dict if successful, None if failed
    """
    try:
        
        if '{' in json_string and '}' in json_string:
            first_brace = json_string.find('{')
            last_brace = json_string.rfind('}')
            if last_brace > first_brace:
                json_string = json_string[first_brace:last_brace + 1]
        return json.loads(json_string)
    except json.JSONDecodeError as e:
        logger.warning(f"Initial JSON parse failed: {e}")
        

        fixed_json = json_string
        
        # Remove trailing commas
        fixed_json = re.sub(r',\s*}', '}', fixed_json)
        fixed_json = re.sub(r',\s*]', ']', fixed_json)
        
        # FFix unquoted property names
        fixed_json = re.sub(r'(\w+):', r'"\1":', fixed_json)
        
        # Fix single quotes to double quotes
        fixed_json = re.sub(r"'([^']*)':", r'"\1":', fixed_json)
        
        # Trim to the outermost braces if present
        first_brace = fixed_json.find('{')
        last_brace = fixed_json.rfind('}')
        if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
            fixed_json = fixed_json[first_brace:last_brace + 1]

        # Escape raw newlines and tabs so long text stays valid inside strings
        fixed_json = fixed_json.replace('\r\n', '\\n').replace('\n', '\\n').replace('\t', ' ')
        # Remove remaining invalid control characters
        fixed_json = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', ' ', fixed_json)
        
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

    not_found_json = '{"hotel": null, "rooms": null, "amenities": null, "policies": null, "special_features": null, "awards": null, "error": "Hotel doesn\'t exist"}'
    header = (
        "You are a hotel data researcher. Gather comprehensive information about:\n\n"
        f"**Hotel Name:** {hotel_name}\n"
        f"**Location:** {location}\n\n"
        f"First verify the hotel exists at this location. If not, return: {not_found_json}\n\n"
        "**OUTPUT RULES:**\n"
        "- Return EXACTLY ONE JSON code block (no text before/after)\n"
        "- Single JSON object (no top-level arrays)\n"
        "- Escape newlines as \\n\n"
        "- Use null for missing data\n"
        "- All data types must match schema below\n\n"
        "**REQUIRED DATA:**\n\n"
        "1. **Hotel Main Info:**\n"
        "   - official_name, star_rating (1-5), description (2-3 paragraphs, no bullets), address, year_opened (YYYY-MM-DD), last_renovation (YYYY-MM-DD), total_rooms, number_of_floors, additional_info (2-3 paragraphs, no bullets)\n\n"
        "2. **Room Types:**\n"
        "   - room_type, bed_configuration, room_size_sqft (integer only), max_occupancy, price_range_min/max (decimal XX.XX), description\n\n"
        "3. **Amenities by Category:**\n"
        "   - Room Amenities, Property Amenities, Dining, Services\n"
        "   - Each with: category, description, details (JSON object)\n\n"
        "4. **Policies:**\n"
        "   - check_in_time (HH:MM:SS), check_out_time (HH:MM:SS), min_age_requirement, pet_policy, smoking_policy, children_policy, extra_person_policy, cancellation_policy\n\n"
        "5. **Special Features & Awards:**\n"
        "   - Arrays of strings\n\n"
        "**SCHEMA:**\n"
    )
    schema_block = (
        "```json\n"
        "{\n"
        "  \"hotel\": {\n"
        "    \"official_name\": \"string (max 255)\",\n"
        "    \"star_rating\": 4,\n"
        "    \"description\": \"2-3 paragraphs prose, no bullets\",\n"
        "    \"address\": \"string\",\n"
        "    \"year_opened\": \"2015-01-01\",\n"
        "    \"last_renovation\": \"2021-01-01\",\n"
        "    \"total_rooms\": 261,\n"
        "    \"number_of_floors\": 5,\n"
        "    \"additional_info\": \"2-3 paragraphs prose, no bullets\"\n"
        "  },\n"
        "  \"rooms\": [\n"
        "    {\n"
        "      \"room_type\": \"string (max 100)\",\n"
        "      \"bed_configuration\": \"string (max 100)\",\n"
        "      \"room_size_sqft\": 280,\n"
        "      \"max_occupancy\": 2,\n"
        "      \"price_range_min\": 189.00,\n"
        "      \"price_range_max\": 279.00,\n"
        "      \"description\": \"string\"\n"
        "    }\n"
        "  ],\n"
        "  \"amenities\": [\n"
        "    {\n"
        "      \"category\": \"string (max 50)\",\n"
        "      \"description\": \"string\",\n"
        "      \"details\": {}\n"
        "    }\n"
        "  ],\n"
        "  \"policies\": {\n"
        "    \"check_in_time\": \"15:00:00\",\n"
        "    \"check_out_time\": \"11:00:00\",\n"
        "    \"min_age_requirement\": 21,\n"
        "    \"pet_policy\": \"string\",\n"
        "    \"smoking_policy\": \"string\",\n"
        "    \"children_policy\": \"string\",\n"
        "    \"extra_person_policy\": \"string\",\n"
        "    \"cancellation_policy\": \"string\"\n"
        "  },\n"
        "  \"special_features\": [\"string\"],\n"
        "  \"awards\": [\"string\"]\n"
        "}\n"
        "```\n\n"
        "Prioritize: Official hotel website, Expedia, Booking.com, TripAdvisor.\n\n"
        "Begin your search and provide the complete JSON response matching the exact schema structure above."
    )

    prompt = header + schema_block

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

        print(content)
        
        # Extract JSON from markdown code blocks (support multiple blocks)
        if '```' in content:
            try:
                blocks = re.findall(r"```(?:json)?\s*([\s\S]*?)```", content, flags=re.IGNORECASE)
            except Exception:
                blocks = []
            if blocks:
                
                blocks_sorted = sorted(blocks, key=lambda b: len(b or ''), reverse=True)
                content = (blocks_sorted[0] or '').strip()
            else:
               
                if '{' in content and '}' in content:
                    first_brace = content.find('{')
                    last_brace = content.rfind('}')
                    if last_brace > first_brace:
                        content = content[first_brace:last_brace + 1]
        else:
   
            if '{' in content and '}' in content:
                first_brace = content.find('{')
                last_brace = content.rfind('}')
                if last_brace > first_brace:
                    content = content[first_brace:last_brace + 1]
        
        # Log the extracted content for debugging
        logger.debug(f"Extracted content for {hotel_name}: {content[:200]}...")
        
        # Handle explicit non-JSON responses
        if content.strip().lower().startswith("hotel doesn't exist"):
            return {
                'hotel': None,
                'rooms': None,
                'amenities': None,
                'policies': None,
                'special_features': None,
                'awards': None,
                'error': "Hotel doesn't exist"
            }

        # Validate JSON before parsing
        if not content.strip():
            logger.error(f"Empty content extracted for hotel: {hotel_name}")
            return None
            
        hotel_data = validate_and_fix_json(content)
        if hotel_data:
            logger.info(f"Successfully retrieved data for hotel: {hotel_name}")
            return hotel_data
        else:
            print(content)
            logger.error(f"Failed to parse JSON for hotel: {hotel_name}")
            return None
        
    except Exception as e:
        logger.error(f"Failed to retrieve data for hotel {hotel_name}: {str(e)}")
        return None


def extract_hotel_data_from_row(df_row: pd.Series) -> Dict:
    hotel_name = df_row.get('name', '')
    street = df_row.get('address_street-address', '')
    city = df_row.get('address_locality', '')
    state = df_row.get('address_region', '')
    postal_code = df_row.get('address_postal-code', '')
    if not hotel_name:
        raise ValueError("Hotel name not found in DataFrame row")
   
    parts = [str(p).strip() for p in [street, city, state] if p and str(p).strip()]
    # Attach postal code with a space before it if present
    full_address = ", ".join(parts)
    if postal_code and str(postal_code).strip():
        full_address = f"{full_address} {str(postal_code).strip()}" if full_address else str(postal_code).strip()
    if not full_address:
        raise ValueError("Location not found in DataFrame row")
    return extract_hotel_data(hotel_name, full_address)


def safe_get(source, key, default=None):
        try:
            return source.get(key, default) if hasattr(source, 'get') else default
        except:
            return default

def safe_int(value, default=None):
    """Convert to integer with default"""
    try:
        if pd.isna(value) or value is None or str(value).strip() in ('', 'NaN', 'nan'):
            return default
        return int(float(value))
    except:
        return default

def safe_decimal(value, default=None):
    """Convert to decimal/float with 2 decimal places"""
    try:
        if pd.isna(value) or value is None or str(value).strip() in ('', 'NaN', 'nan'):
            return default
        result = float(value)
        return round(result, 2)
    except:
        return default

def safe_star_rating(value):
    """Ensure star rating is between 1-5 or None"""
    rating = safe_int(value)
    if rating is None:
        return None
    return max(1, min(5, rating))  # Clamp between 1-5

def safe_date(value):
    """Convert to DATE format"""
    if pd.isna(value) or value is None:
        return None
    try:
        date_obj = pd.to_datetime(value, errors='coerce')
        return None if pd.isna(date_obj) else date_obj.strftime('%Y-%m-%d')
    except:
        return None

def safe_jsonb(value):
    """Convert to JSONB-compatible JSON string or None"""
    print("Value", value,pd.isna(value) )
    try:
        if pd.isna(value) or value is None:
            return None
        if isinstance(value, str):
            value = value.strip()
            if value in ('', 'NaN', 'nan'):
                return None
            json.loads(value)
            return value
        return json.dumps(value)
    except:
        return None
        
def merge_hotel_data(df_row: pd.Series, hotel_data_json: Dict) -> Dict[str, pd.DataFrame]:
    # Safely handle None inputs
    if df_row is None:
        df_row = pd.Series()
    if hotel_data_json is None:
        hotel_data_json = {}
    
    hotel_info = hotel_data_json.get('hotel', {}) if hotel_data_json else {}
    print("safe_date(safe_get(hotel_info, 'year_opened')),",safe_date(safe_get(hotel_info, 'year_opened')))
    hotels_data = {
        'hotel_id': safe_get(df_row, 'id'),
        'official_name': (
            safe_get(hotel_info, 'official_name') or 
            safe_get(df_row, 'name') or 
            'Hotel Name Not Available'
        ),
        'star_rating': safe_star_rating(
            safe_get(hotel_info, 'star_rating') or safe_get(df_row, 'hotel_class')
        ),
        'description': safe_get(hotel_info, 'description'),
        'address': safe_get(df_row, 'address_street-address'),
        'additional_info': safe_get(hotel_info, 'additional_info'),
        'city': safe_get(df_row, 'address_locality'),
        'state': safe_get(df_row, 'address_region'),
        'zip_code': safe_get(df_row, 'address_postal-code'),
        'country': 'USA',
        'phone': safe_get(df_row, 'phone'),
        'email': None,
        'website': safe_get(df_row, 'url'),
        'overall_rating': None,
        'cleanliness_rating': None,
        'service_rating': None,
        'location_rating': None,
        'value_rating': None,
        'total_reviews': 0,
        'total_rooms': safe_int(safe_get(hotel_info, 'total_rooms')),
        'number_of_floors': safe_int(safe_get(hotel_info, 'number_of_floors')),
        'year_opened': safe_date(safe_get(hotel_info, 'year_opened')),
        'last_renovation': safe_date(safe_get(hotel_info, 'last_renovation')),
        'images': safe_jsonb(safe_get(hotel_info, 'images'))
    }

    rooms_list = []
    try:
        rooms_list = (hotel_data_json or {}).get('rooms', []) or []
    except:
        pass

    rooms_data: List[Dict] = []
    for room in rooms_list:
        if not room:
            continue
        try:
            rooms_data.append({
                'hotel_id': safe_get(df_row, 'id'),
                'room_type': safe_get(room, 'room_type') or 'Standard Room',
                'bed_configuration': safe_get(room, 'bed_configuration'),
                'room_size_sqft': safe_int(safe_get(room, 'room_size_sqft')),
                'max_occupancy': safe_int(safe_get(room, 'max_occupancy')),
                'price_range_min': safe_decimal(safe_get(room, 'price_range_min')),
                'price_range_max': safe_decimal(safe_get(room, 'price_range_max')),
                'description': safe_get(room, 'description')
            })
        except:
            continue

    # AMENITIES TABLE DATA
    amenities_list = []
    try:
        amenities_list = (hotel_data_json or {}).get('amenities', []) or []
    except:
        pass

    amenities_data: List[Dict] = []
    for amenity in amenities_list:
        if not amenity:
            continue
        try:
            details = safe_get(amenity, 'details')
            amenities_data.append({
                'hotel_id': safe_get(df_row, 'id'),
                'category': safe_get(amenity, 'category'),
                'description': safe_get(amenity, 'description'),
                # Convert to TEXT (JSON string) if it's a dict/list
                'details': json.dumps(details) if isinstance(details, (dict, list)) else str(details) if details else None
            })
        except:
            continue

    # POLICIES TABLE DATA
    policies_info = {}
    try:
        policies_info = (hotel_data_json or {}).get('policies', {}) or {}
    except:
        pass

    policies_df = pd.DataFrame()
    try:
        if policies_info and any(policies_info.values()):
            policies_df = pd.DataFrame([{
                'hotel_id': safe_get(df_row, 'id'),
                'check_in_time': safe_get(policies_info, 'check_in_time'),
                'check_out_time': safe_get(policies_info, 'check_out_time'),
                'min_age_requirement': safe_int(safe_get(policies_info, 'min_age_requirement')),
                'pet_policy': safe_get(policies_info, 'pet_policy'),
                'smoking_policy': safe_get(policies_info, 'smoking_policy'),
                'children_policy': safe_get(policies_info, 'children_policy'),
                'extra_person_policy': safe_get(policies_info, 'extra_person_policy'),
                'cancellation_policy': safe_get(policies_info, 'cancellation_policy')
            }])
    except:
        policies_df = pd.DataFrame()

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
            df.to_csv(os.path.join(output_dir, f'batch_{table}.csv'), index=False)

