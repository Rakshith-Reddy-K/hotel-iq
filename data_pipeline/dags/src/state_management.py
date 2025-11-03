import json
import logging
from datetime import datetime
import pandas as pd

from src.bucket_util import download_file_from_gcs, upload_file_to_gcs
from src.path import (
    get_processed_batch_file_path,
    get_processed_state_path,
    get_gcs_processed_state_path
)

logger = logging.getLogger(__name__)


def update_processing_state(city: str = 'Boston') -> str:
    logger.info(f"Updating processing state for {city}")
    
    try:
        batch_path = get_processed_batch_file_path(city, 'hotels')
        batch = pd.read_csv(batch_path)
        hotel_ids = batch['hotel_id'].tolist()
        
        state_path = get_processed_state_path(city)
        try:
            download_file_from_gcs(get_gcs_processed_state_path(city), state_path)
            with open(state_path, 'r') as f:
                state = json.load(f)
        except Exception:
            state = {
                "city": city,
                "processed_hotel_ids": [],
                "total_processed": 0,
                "batches": []
            }
        state['processed_hotel_ids'].extend(hotel_ids)
        state['total_processed'] = len(state['processed_hotel_ids'])
        state['last_updated'] = datetime.now().isoformat()
        state['batches'].append({
            "batch_id": len(state['batches']) + 1,
            "hotel_ids": hotel_ids,
            "count": len(hotel_ids),
            "processed_at": datetime.now().isoformat(),
            "status": "completed"
        })
        
        with open(state_path, 'w') as f:
            json.dump(state, f, indent=2)
        
        upload_file_to_gcs(state_path, get_gcs_processed_state_path(city))
        
        logger.info(f"State updated! Total processed: {state['total_processed']} hotels")
        logger.info(f"Completed batch {len(state['batches'])}")
        
        return state_path
        
    except Exception as e:
        logger.error(f"Failed to update state: {str(e)}")
        raise