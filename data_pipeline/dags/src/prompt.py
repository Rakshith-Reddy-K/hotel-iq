import os
import time
import json
import pandas as pd

from src.path import (
    get_batch_hotels_path,
    get_batch_enrichment_path,
)
from typing import Dict, Optional
import logging
import re
import requests

logger = logging.getLogger(__name__)

def extract_hotel_data_gemini(hotel_name: str, location: str, api_key: Optional[str] = None) -> Optional[Dict]:
    """
    Extract comprehensive hotel data using Gemini API (default).
    Uses utils_gemini.get_hotel_data directly.
    
    Args:
        hotel_name (str): Name of the hotel
        location (str): Location of the hotel (e.g., "Boston, MA 02110")
        api_key (ignored): Kept for backward compatibility; Gemini uses env var GEMINI_API_KEY.
        
    Returns:
        dict: JSON response containing detailed hotel information, or None on error
    """
    from src.utils_gemini import get_hotel_data
    return get_hotel_data(hotel_name, location)

def extract_hotel_data_perplexity(hotel_name: str, location: str, api_key: Optional[str] = None) -> Optional[Dict]:
    """
    Extract comprehensive hotel data using Perplexity API (legacy).
    
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
        
        logger.debug(f"Extracted content for {hotel_name}: {content[:200]}...")
        
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

def validate_and_fix_json(json_string: str) -> Optional[Dict]:
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

def extract_hotel_data_from_row_gemini(df_row: pd.Series) -> Dict:
    """
    Extract hotel data from a DataFrame row using Gemini API (default).
    """
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
    return extract_hotel_data_gemini(hotel_name, full_address)

def extract_hotel_data_from_row_perplexity(df_row: pd.Series) -> Dict:
    """
    Extract hotel data from a DataFrame row using Perplexity API (legacy).
    """
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
    return extract_hotel_data_perplexity(hotel_name, full_address)

def enrich_hotels_perplexity(city: str = 'Boston', delay_seconds: float = 12, max_hotels: int = None):
    """
    Enrich hotels using Perplexity API (legacy).
    """
    hotels_csv = get_batch_hotels_path(city)
    if not os.path.exists(hotels_csv):
        raise FileNotFoundError(f"Expected hotels CSV not found: {hotels_csv}")

    df = pd.read_csv(hotels_csv)
    if max_hotels is not None:
        df = df.head(max_hotels)

    enrichment_path = get_batch_enrichment_path(city)

    with open(enrichment_path, 'w', encoding='utf-8') as f:
        for _, row in df.iterrows():
            try:
                data = extract_hotel_data_from_row_perplexity(row)
                record = {
                    'hotel_id': row.get('id'),
                    'name': row.get('name'),
                    'city': row.get('address_locality'),
                    'data': data
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
            except Exception as e:
                err_record = {
                    'hotel_id': row.get('id'),
                    'name': row.get('name'),
                    'error': str(e)
                }
                f.write(json.dumps(err_record, ensure_ascii=False) + "\n")
            time.sleep(delay_seconds)

    return enrichment_path

def enrich_hotels_gemini(
    city: str = 'Boston',
    delay_seconds: float = 12,
    max_hotels: int = None
):
    """
    Enrich hotels using Gemini API (default).
    """
    hotels_csv = get_batch_hotels_path(city)
    if not os.path.exists(hotels_csv):
        raise FileNotFoundError(f"Expected hotels CSV not found: {hotels_csv}")

    df = pd.read_csv(hotels_csv)
    if max_hotels is not None:
        df = df.head(max_hotels)

    enrichment_path = get_batch_enrichment_path(city)

    with open(enrichment_path, 'w', encoding='utf-8') as f:
        for _, row in df.iterrows():
            try:
                data = extract_hotel_data_from_row_gemini(row)
                record = {
                    'hotel_id': row.get('id'),
                    'name': row.get('name'),
                    'city': row.get('address_locality'),
                    'data': data
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
            except Exception as e:
                err_record = {
                    'hotel_id': row.get('id'),
                    'name': row.get('name'),
                    'error': str(e)
                }
                f.write(json.dumps(err_record, ensure_ascii=False) + "\n")
            time.sleep(delay_seconds)

    return enrichment_path