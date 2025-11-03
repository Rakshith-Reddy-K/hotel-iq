import os
import json
import logging
import pandas as pd

from src.bucket_util import download_file_from_gcs, upload_file_to_gcs
from src.path import (
    get_filtered_hotels_path,
    get_batch_hotels_path,
    get_batch_reviews_path,
    get_filtered_reviews_path,
    get_processed_state_path,
    get_gcs_filtered_hotels_path,
    get_gcs_filtered_reviews_path,
    get_gcs_processed_batch_hotels_path,
    get_gcs_processed_batch_reviews_path,
    get_gcs_processed_state_path
)

logger = logging.getLogger(__name__)


def select_next_batch(city: str = 'Boston', batch_size: int = 50) -> str:
    logger.info(f"Selecting next batch for {city} (size: {batch_size})")
    
    try:
        filtered_hotels_path = get_filtered_hotels_path(city)
        download_file_from_gcs(get_gcs_filtered_hotels_path(city), filtered_hotels_path)
        all_hotels = pd.read_csv(filtered_hotels_path)
        logger.info(f"Loaded {len(all_hotels)} total hotels for {city}")
        
        state_path = get_processed_state_path(city)
        try:
            download_file_from_gcs(get_gcs_processed_state_path(city), state_path)
            with open(state_path, 'r') as f:
                state = json.load(f)
            processed_ids = set(state['processed_hotel_ids'])
            logger.info(f"Already processed: {len(processed_ids)} hotels")
        except Exception:
            state = {
                "city": city,
                "processed_hotel_ids": [],
                "total_processed": 0,
                "batches": []
            }
            processed_ids = set()
            logger.info("Starting fresh - no hotels processed yet")
        
        # Find unprocessed hotels
        unprocessed = all_hotels[~all_hotels['id'].isin(processed_ids)]
        
        if len(unprocessed) == 0:
            logger.info("ALL HOTELS PROCESSED!")
            raise Exception("All hotels already processed")
        
        batch = unprocessed.head(batch_size)
        logger.info(f"Selected {len(batch)} hotels for this batch")
        logger.info(f"Remaining: {len(unprocessed) - len(batch)} hotels")
        
        batch_path = get_batch_hotels_path(city)
        batch.to_csv(batch_path, index=False)
        upload_file_to_gcs(batch_path, get_gcs_processed_batch_hotels_path(city))
        
        logger.info(f"Batch saved to: {batch_path}")
        return batch_path
        
    except Exception as e:
        logger.error(f"Failed to select batch: {str(e)}")
        raise


def filter_reviews_for_batch(city: str = 'Boston') -> str:
    logger.info("Filtering reviews for current batch")
    try:
        batch_path = get_batch_hotels_path(city)
        batch = pd.read_csv(batch_path)
        hotel_ids = batch['id'].tolist()
        logger.info(f"Filtering reviews for {len(hotel_ids)} hotels in batch")
        
        city_reviews_path = get_filtered_reviews_path(city)
        download_file_from_gcs(get_gcs_filtered_reviews_path(city), city_reviews_path)
        all_city_reviews = pd.read_csv(city_reviews_path)
        logger.info(f"Loaded {len(all_city_reviews)} cached city reviews")
        
        batch_reviews = all_city_reviews[all_city_reviews['hotel_id'].isin(hotel_ids)]
        logger.info(f"Found {len(batch_reviews)} reviews for this batch")
        
        batch_reviews_path = get_batch_reviews_path(city)
        batch_reviews.to_csv(batch_reviews_path, index=False)
        upload_file_to_gcs(batch_reviews_path, get_gcs_processed_batch_reviews_path(city))
        
        logger.info(f"Batch reviews saved to: {batch_reviews_path}")
        return batch_reviews_path
        
    except Exception as e:
        logger.error(f"Failed to filter batch reviews: {str(e)}")
        raise