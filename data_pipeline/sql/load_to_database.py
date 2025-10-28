import os
import csv
import pandas as pd
import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import psycopg2
from dotenv import load_dotenv
from db_pool import db_pool

# Load environment variables
load_dotenv()

# Configuration
LOCAL_CSV_PATH = os.getenv("LOCAL_CSV_PATH", "data_pipeline/intermediate/csv")

# ============= CONNECTION MANAGEMENT =============

def get_db_connection():
    """Get direct psycopg2 connection"""
    return db_pool.get_connection()

def import_csv_with_copy(
    csv_file_path: str,
    table_name: str,
    columns: List[str],
    delimiter: str = ',',
    null_string: str = '',
    has_header: bool = True
) -> int:
    """
    Import CSV using PostgreSQL COPY command (fastest method)
    
    Args:
        csv_file_path: Path to local CSV file
        table_name: Target table name
        columns: List of column names
        delimiter: CSV delimiter
        null_string: String representing NULL values
        has_header: Whether CSV has header row
    
    Returns:
        Number of rows imported
    """
    with get_db_connection() as conn:
        cur = conn.cursor()
    
        try:
            logging.info(f"📂 Importing CSV with COPY: {csv_file_path}")
            
            with open(csv_file_path, 'r', encoding='utf-8') as f:
                # Skip header if present
                if has_header:
                    next(f)
                
                # Use COPY command for bulk loading
                cur.copy_expert(
                    sql=f"""
                        COPY {table_name} ({','.join(columns)})
                        FROM STDIN
                        WITH (
                            FORMAT CSV,
                            DELIMITER '{delimiter}',
                            NULL '{null_string}',
                            QUOTE '"',
                            ESCAPE '\\'
                        )
                    """,
                    file=f
                )
            
            rows_imported = cur.rowcount
            conn.commit()
            
            logging.info(f"Imported {rows_imported} rows to {table_name}")
            return rows_imported
            
        except Exception as e:
            conn.rollback()
            logging.error(f" Error in COPY import: {e}")
            raise

# ============= SPECIFIC TABLE IMPORT FUNCTIONS =============

def import_hotels_from_local(csv_file_path: str) -> int:
    """
    Import hotels data from local CSV
    
    Args:
        csv_file_path: Path to hotels CSV file
    """
    columns = [
    'hotel_id',
    'official_name',
    'star_rating',
    'description',
    'address',
    'city',
    'state',
    'zip_code',
    'country',
    'phone',
    'email',
    'website',
    'overall_rating',
    'total_reviews',
    'cleanliness_rating',
    'service_rating',
    'location_rating',
    'value_rating',
    'year_opened',
    'last_renovation',
    'total_rooms',
    'number_of_floors',
    'additional_info'
    ]
    
    return import_csv_with_copy(csv_file_path, 'hotels', columns)

def import_rooms_from_local(csv_file_path: str) -> int:
    """Import rooms data from local CSV"""
    columns = [
    'hotel_id',
    'room_type',
    'bed_configuration',
    'room_size_sqft',
    'max_occupancy',
    'price_range_min',
    'price_range_max',
    'description'
    ]
   
    return import_csv_with_copy(csv_file_path, 'rooms', columns)

def import_reviews_from_local(csv_file_path: str) -> int:
    """Import reviews data from local CSV"""
    columns = [
        'hotel_id', 'overall_rating', 'title',
        'review_text', 'reviewer_name',
        'review_date', 'source'
    ]
    
    
    return import_csv_with_copy(csv_file_path, 'reviews', columns)
   

def import_amenities_from_local(csv_file_path: str) -> int:
    """Import amenities data from local CSV"""
    columns = ['hotel_id', 'category', 'description', 'details']
    
   
    return import_csv_with_copy(csv_file_path, 'amenities', columns)
   

def import_policies_from_local(csv_file_path: str) -> int:
    """Import policies data from local CSV"""
    columns = [
        'hotel_id', 'check_in_time', 'check_out_time',
        'min_age_requirement', 'pet_policy', 'smoking_policy',
        'children_policy', 'extra_person_policy', 'cancellation_policy'
    ]

    return import_csv_with_copy(csv_file_path, 'policies', columns)
   
def load_all_hotel_data_to_database(
    csv_directory: str = "data_pipeline/data/csv",
) -> Dict[str, int]:
    """
    Import all hotel data from local CSV files
    
    Args:
        csv_directory: Directory containing CSV files
    
    Returns:
        Dictionary with import results
    """
    from pathlib import Path
    
    csv_dir = Path(csv_directory)
    results = {}
    
    # Check if directory exists
    if not csv_dir.exists():
        raise FileNotFoundError(f"Directory not found: {csv_directory}")
    
    logging.info("="*60)
    logging.info("IMPORTING FROM LOCAL CSV FILES")
    logging.info(f"   Directory: {csv_directory}")
    logging.info("="*60)
    
    # Import order (respect foreign key constraints)
    import_order = [
        ('hotels', 'hotels.csv', import_hotels_from_local),
        ('rooms', 'rooms.csv', import_rooms_from_local),
        ('reviews', 'reviews.csv', import_reviews_from_local),
        ('amenities', 'amenities.csv', import_amenities_from_local),
        ('policies', 'policies.csv', import_policies_from_local)
    ]
    
    for table_name, csv_filename, import_func in import_order:
        csv_path = csv_dir / csv_filename
        
        if not csv_path.exists():
            logging.warning(f"⚠️ File not found: {csv_path}")
            results[table_name] = 0
            continue
        
        try:
            logging.info(f"\n📊 Importing {table_name}...")
            rows = import_func(str(csv_path))
            results[table_name] = rows
            
        except Exception as e:
            logging.error(f" Failed to import {table_name}: {e}")
            results[table_name] = -1
    
    # Summary
    logging.info("\n" + "="*60)
    logging.info("📊 IMPORT SUMMARY")
    logging.info("="*60)
    
    for table, rows in results.items():
        if rows >= 0:
            logging.info(f"{table}: {rows} rows")
        else:
            logging.info(f" {table}: Failed")
    
    return results