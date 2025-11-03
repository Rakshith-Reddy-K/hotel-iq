import os
import logging
import shutil
import pandas as pd

from src.bucket_util import download_file_from_gcs, upload_file_to_gcs
from src.path import (
    TABLE_NAMES,
    get_processed_batch_file_path,
    get_processed_table_file_path,
    get_processed_enrichment_path,
    get_gcs_processed_table_path,
    get_gcs_processed_enrichment_path
)

logger = logging.getLogger(__name__)


def append_batch_to_accumulated(city: str = 'Boston') -> dict:
    logger.info(f"Appending batch to accumulated results for {city}")
    
    try:
        accumulated_paths = {}
        
        for table in TABLE_NAMES:
            batch_file = get_processed_batch_file_path(city, table)
            
            if not os.path.exists(batch_file):
                continue
            
            batch_df = pd.read_csv(batch_file)
            accumulated_file = get_processed_table_file_path(city, table)
            
            try:
                download_file_from_gcs(get_gcs_processed_table_path(city, table), accumulated_file)
                accumulated_df = pd.read_csv(accumulated_file)
                combined_df = pd.concat([accumulated_df, batch_df], ignore_index=True)
            except:
                combined_df = batch_df
            
            if table == 'hotels' and 'hotel_id' in combined_df.columns:
                combined_df = combined_df.drop_duplicates(subset=['hotel_id'], keep='last')
            
            combined_df.to_csv(accumulated_file, index=False)
            upload_file_to_gcs(accumulated_file, get_gcs_processed_table_path(city, table))
            accumulated_paths[table] = accumulated_file
        
        batch_enrichment = get_processed_batch_file_path(city, 'enrichment').replace('.csv', '.jsonl')
        
        if os.path.exists(batch_enrichment):
            accumulated_enrichment = get_processed_enrichment_path(city)
            
            try:
                download_file_from_gcs(get_gcs_processed_enrichment_path(city), accumulated_enrichment)
                with open(accumulated_enrichment, 'a', encoding='utf-8') as acc_file:
                    with open(batch_enrichment, 'r', encoding='utf-8') as batch_file:
                        acc_file.write(batch_file.read())
            except:
                shutil.copy(batch_enrichment, accumulated_enrichment)
            
            upload_file_to_gcs(accumulated_enrichment, get_gcs_processed_enrichment_path(city))
            accumulated_paths['enrichment'] = accumulated_enrichment
        
        logger.info("Successfully appended batch to accumulated files")
        return accumulated_paths
        
    except Exception as e:
        logger.error(f"Failed to append: {e}")
        raise