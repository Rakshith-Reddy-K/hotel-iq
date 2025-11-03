import json
import os
import logging
from typing import Dict, List, Optional, Tuple
from src.bucket_util import download_file_from_gcs
import pandas as pd
from dotenv import load_dotenv
import json

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

