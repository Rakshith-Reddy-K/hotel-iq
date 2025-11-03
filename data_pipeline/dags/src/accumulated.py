import os
import logging
import shutil
from pathlib import Path

import pandas as pd

from src.bucket_util import download_file_from_gcs, upload_file_to_gcs
from src.path import _resolve_project_path
import shutil

logger = logging.getLogger(__name__)
def append_batch_to_accumulated(city: str = 'Boston') -> dict:
    logger.info(f"Appending batch to accumulated results for {city}")
    
    try:
        city_lower = city.lower().replace(' ', '_')
        output_dir = _resolve_project_path(f'data/processed/{city_lower}')
        
        # Append structured CSVs
        tables = ['hotels', 'rooms', 'amenities', 'policies', 'reviews']
        accumulated_paths = {}
        
        for table in tables:
            batch_file = os.path.join(output_dir, f'batch_{table}.csv')
            
            if not os.path.exists(batch_file):
                continue
            
            batch_df = pd.read_csv(batch_file)
            accumulated_file = os.path.join(output_dir, f'{table}.csv')
            
            try:
                download_file_from_gcs(f"processed/{city_lower}/{table}.csv", accumulated_file)
                accumulated_df = pd.read_csv(accumulated_file)
                combined_df = pd.concat([accumulated_df, batch_df], ignore_index=True)
            except:
                combined_df = batch_df
            
            if table == 'hotels' and 'hotel_id' in combined_df.columns:
                combined_df = combined_df.drop_duplicates(subset=['hotel_id'], keep='last')
            
            combined_df.to_csv(accumulated_file, index=False)
            upload_file_to_gcs(accumulated_file, f"processed/{city_lower}/{table}.csv")
            accumulated_paths[table] = accumulated_file
        
        batch_enrichment = os.path.join(output_dir, 'batch_enrichment.jsonl')
        
        if os.path.exists(batch_enrichment):
            accumulated_enrichment = os.path.join(output_dir, 'enrichment.jsonl')
            
            try:
                download_file_from_gcs(f"processed/{city_lower}/enrichment.jsonl", accumulated_enrichment)
                # Append batch to existing
                with open(accumulated_enrichment, 'a', encoding='utf-8') as acc_file:
                    with open(batch_enrichment, 'r', encoding='utf-8') as batch_file:
                        acc_file.write(batch_file.read())
            except:
                # First batch - just copy
                shutil.copy(batch_enrichment, accumulated_enrichment)
            
            upload_file_to_gcs(accumulated_enrichment, f"processed/{city_lower}/enrichment.jsonl")
            accumulated_paths['enrichment'] = accumulated_enrichment
        
        logger.info("Successfully appended batch to accumulated files")
        return accumulated_paths
        
    except Exception as e:
        logger.error(f"Failed to append: {e}")
        raise