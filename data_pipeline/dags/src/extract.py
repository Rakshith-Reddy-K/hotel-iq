import os
import logging
from pathlib import Path

from dotenv import load_dotenv
import pandas as pd

from .utils import load_offering_json, get_sample_hotels_by_city, load_reviews_json
from .bucket_util import upload_file_to_gcs

# Configure logging
logger = logging.getLogger(__name__)
load_dotenv()

def _ensure_output_dir(output_dir: str) -> None:
    Path(output_dir).mkdir(parents=True, exist_ok=True)


def _resolve_project_path(path_str: str) -> str:
    """Resolve a path relative to the project root (two levels up from this file)."""
    path = Path(path_str)
    if path.is_absolute():
        return str(path)
    project_root = Path(__file__).resolve().parents[2]
    return str((project_root / path).resolve())


def extract_metadata(
    city: str = 'Boston',
    sample_size: int = 25,
    random_seed: int = 42,
    offering_path: str = 'data/raw/hotels.txt',
    output_dir: str = 'output'
):
    """
    Extract hotel metadata from offering.txt and sample by city.
    
    Returns:
        str: Path to generated CSV file
    """
    logger.info(f"Starting hotel metadata extraction for city: {city}")
    logger.info(f"Configuration - Sample size: {sample_size}, Random seed: {random_seed}")
    
    try:
        offering_abspath = _resolve_project_path(offering_path)
        logger.info(f"Reading offering data from: {offering_abspath}")
        
        if not os.path.exists(offering_abspath):
            raise FileNotFoundError(f"Offering file not found: {offering_abspath}")
        
        df = load_offering_json(offering_abspath)
        logger.info(f"Successfully loaded {len(df)} total hotels from offering file")
        
        sample_df = get_sample_hotels_by_city(df, city, sample_size=sample_size, random_seed=random_seed)
        logger.info(f"Selected {len(sample_df)} hotels for {city}")
        
        city_token = city.title().replace(' ', '')
        output_abspath = _resolve_project_path(output_dir)
        _ensure_output_dir(output_abspath)
        output_path = os.path.join(output_abspath, f'hotel_data_{city_token}.csv')
        
        sample_df.to_csv(output_path, index=False)
        logger.info(f"Hotel metadata saved to: {output_path}")
        upload_file_to_gcs(output_path, os.getenv('GCS_RAW_HOTELS_DATA_PATH'))
        return output_path
        
    except Exception as e:
        logger.error(f"Failed to extract hotel metadata: {str(e)}")
        raise


def extract_reviews(
    reviews_path: str = 'data/raw/reviews.txt',
    output_dir: str = 'output'
):
    """
    Extract and normalize reviews from review.txt.
    
    Returns:
        str: Path to generated CSV file
    """
    logger.info("Starting reviews extraction and normalization")
    
    try:
        reviews_abspath = _resolve_project_path(reviews_path)
        logger.info(f"Reading reviews from: {reviews_abspath}")
        
        if not os.path.exists(reviews_abspath):
            raise FileNotFoundError(f"Reviews file not found: {reviews_abspath}")
        
        reviews_df = load_reviews_json(reviews_abspath)
        logger.info(f"Successfully loaded {len(reviews_df)} reviews")
        
        output_abspath = _resolve_project_path(output_dir)
        _ensure_output_dir(output_abspath)
        output_path = os.path.join(output_abspath, 'reviews.csv')
        
        reviews_df.to_csv(output_path, index=False)
        logger.info(f"Normalized reviews saved to: {output_path}")
        
        #upload_file_to_gcs(output_path, os.getenv('GCS_RAW_REVIEWS_DATA_PATH'))
        return output_path
        
    except Exception as e:
        logger.error(f"Failed to extract reviews: {str(e)}")
        raise


def join_metadata_and_reviews():
    return
