import os
import logging
import json
import pandas as pd
from dotenv import load_dotenv

from src.utils import parse_raw_hotels
from src.bucket_util import upload_file_to_gcs, download_file_from_gcs
from src.path import (
    get_raw_hotels_path,
    get_raw_reviews_path,
    get_filtered_hotels_path,
    get_filtered_reviews_path,
    get_gcs_filtered_hotels_path,
    get_gcs_filtered_reviews_path
)

logger = logging.getLogger(__name__)
load_dotenv()


def check_if_filtering_needed(city: str) -> str:
    try:
        download_file_from_gcs(
            get_gcs_filtered_hotels_path(city),
            get_filtered_hotels_path(city)
        )
        download_file_from_gcs(
            get_gcs_filtered_reviews_path(city),
            get_filtered_reviews_path(city)
        )
        
        logger.info(f"Filtered data exists for {city} - skipping filtering!")
        return 'skip_filtering'
        
    except Exception as e:
        logger.info(f"Filtered data not found for {city} - need to filter!")
        return 'do_filtering'


def filter_all_city_hotels(city: str = 'Boston', all_hotels_path: str = 'data/raw/hotels.txt') -> str:
    logger.info(f"Starting hotel filtering for city: {city}")
    
    try:
        # Download raw hotels from GCS
        hotels_abspath = get_raw_hotels_path()
        download_file_from_gcs(os.getenv('GCS_RAW_HOTELS_DATA_PATH'), hotels_abspath)
        
        if not os.path.exists(hotels_abspath):
            raise FileNotFoundError(f"Hotels file not found: {hotels_abspath}")
        
        # Load and filter by city
        df = parse_raw_hotels(hotels_abspath)
        logger.info(f"Successfully loaded {len(df)} total hotels")
        
        city_df = df[df['address_locality'].str.contains(city, case=False, na=False)]
        logger.info(f"Filtered to {len(city_df)} hotels in {city}")
        
        # Save filtered hotels (directory created automatically!)
        output_path = get_filtered_hotels_path(city)
        city_df.to_csv(output_path, index=False)
        logger.info(f"Filtered hotels saved to: {output_path}")
        
        # Upload to GCS
        upload_file_to_gcs(output_path, get_gcs_filtered_hotels_path(city))
        
        return output_path
        
    except Exception as e:
        logger.error(f"Failed to filter city hotels: {str(e)}")
        raise


def filter_all_city_reviews(city: str = 'Boston', all_reviews_path: str = 'data/raw/reviews.txt') -> str:
    logger.info(f"Starting review filtering for city: {city}")
    
    try:
        # Download raw reviews from GCS
        reviews_abspath = get_raw_reviews_path()
        download_file_from_gcs(os.getenv('GCS_RAW_REVIEWS_DATA_PATH'), reviews_abspath)
        
        if not os.path.exists(reviews_abspath):
            raise FileNotFoundError(f"Reviews file not found: {reviews_abspath}")
        
        # Load city hotels to get hotel IDs
        city_hotels_path = get_filtered_hotels_path(city)
        if not os.path.exists(city_hotels_path):
            raise FileNotFoundError("Must run filter_all_city_hotels first!")
        
        city_hotels = pd.read_csv(city_hotels_path)
        hotel_ids = set(city_hotels['id'].tolist())
        logger.info(f"Filtering reviews for {len(hotel_ids)} hotels")
        
        # Process JSONL file line by line
        filtered_reviews = []
        line_count = 0

        with open(reviews_abspath, 'r', encoding='utf-8') as f:
            for line in f:
                line_count += 1
                if line_count % 100000 == 0:
                    logger.info(f"Processed {line_count} lines, found {len(filtered_reviews)} matching reviews...")
                
                line = line.strip()
                if not line:
                    continue
                
                try:
                    review = json.loads(line)
                    if review.get('offering_id') in hotel_ids:
                        filtered_reviews.append(review)
                except json.JSONDecodeError:
                    continue

        logger.info(f"Found {len(filtered_reviews)} reviews for {city}")

        # Convert to DataFrame
        if filtered_reviews:
            from src.utils import parse_raw_reviews
            
            # Temp file path (directory created automatically!)
            temp_jsonl = get_filtered_reviews_path(city).replace('reviews.csv', 'temp_reviews.jsonl')
            
            with open(temp_jsonl, 'w', encoding='utf-8') as f:
                for review in filtered_reviews:
                    f.write(json.dumps(review) + '\n')
            
            all_city_reviews = parse_raw_reviews(temp_jsonl)
            os.remove(temp_jsonl)
        else:
            all_city_reviews = pd.DataFrame()
        
        # Save filtered reviews (directory created automatically!)
        output_path = get_filtered_reviews_path(city)
        all_city_reviews.to_csv(output_path, index=False)
        logger.info(f"Filtered reviews saved to: {output_path}")
        
        # Upload to GCS
        upload_file_to_gcs(output_path, get_gcs_filtered_reviews_path(city))
        logger.info(f"Uploaded to GCS: {get_gcs_filtered_reviews_path(city)}")
        
        return output_path
        
    except Exception as e:
        logger.error(f"Failed to filter city reviews: {str(e)}")
        raise