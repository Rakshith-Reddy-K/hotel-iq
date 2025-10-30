import os
import csv
import pandas as pd
import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from dotenv import load_dotenv
from sql.db_pool import get_connection
from psycopg2.extras import execute_batch

# Load environment variables
load_dotenv()

# Configuration
LOCAL_CSV_PATH = os.getenv("LOCAL_CSV_PATH", "intermediate/csv")

# ============= CONNECTION MANAGEMENT =============

def get_db_connection():
    """Get direct psycopg2 connection"""
    return get_connection()

def batch_upsert_csv(
    csv_file_path: str,
    table_name: str,
    columns: List[str],
    conflict_columns: List[str],
    update_columns: Optional[List[str]] = None,
    batch_size: int = 1000
) -> int:
    """
    Batch UPSERT from CSV file to PostgreSQL table
    
    Args:
        conn: PostgreSQL connection object
        csv_file_path: Path to CSV file
        table_name: Target table name
        columns: List of column names matching CSV
        conflict_columns: Columns for ON CONFLICT (e.g., ['id'])
        update_columns: Columns to update on conflict (None = all except conflict columns)
        batch_size: Number of rows per batch
    
    Returns:
        Number of rows processed
    """
    with get_db_connection() as conn:
        cur = conn.cursor()
    
        # Determine columns to update on conflict
        if update_columns is None:
            update_columns = [col for col in columns if col not in conflict_columns]

        # Build UPSERT query
        placeholders = ','.join(['%s'] * len(columns))
        update_set = ','.join([f"{col} = EXCLUDED.{col}" for col in update_columns])

        query = f"""
            INSERT INTO {table_name} ({','.join(columns)})
            VALUES ({placeholders})
            ON CONFLICT ({','.join(conflict_columns)})
            DO UPDATE SET {update_set}
        """

        rows_processed = 0
        batch = []

        try:
            with open(csv_file_path, 'r') as f:
                reader = csv.reader(f)
                next(reader)  # Skip header
                
                for row in reader:
                    batch.append(row)
                    
                    if len(batch) >= batch_size:
                        execute_batch(cur, query, batch)
                        rows_processed += len(batch)
                        batch = []
                
                # Process remaining rows
                if batch:
                    execute_batch(cur, query, batch)
                    rows_processed += len(batch)
            
            conn.commit()
            logging.info(f"Processed {rows_processed} rows")
            return rows_processed
            
        except Exception as e:
            conn.rollback()
            logging.error(f"Error: {e}")
            raise


def batch_upsert_csv_auto(
    csv_file_path: str,
    table_name: str,
    conflict_columns: List[str],
    update_columns: Optional[List[str]] = None,
    batch_size: int = 1000
) -> int:
    """
    Batch UPSERT with automatic column detection from CSV header
    """
    with get_db_connection() as conn:
        cur = conn.cursor()
        
        # Get column types from database schema
        query = """
            SELECT column_name, data_type
            FROM information_schema.columns
            WHERE table_name = %s
            AND table_schema = 'public'
        """
        
        cur.execute(query, (table_name,))
        results = cur.fetchall()
        
        logging.info(f"\n{'='*60}")
        logging.info(f"Table: {table_name}")
        logging.info(f"Database query returned {len(results)} columns")
        
        column_types = {row[0]: row[1] for row in results}
        
        # Debug: print all database columns
        logging.info(f"Database columns: {sorted(column_types.keys())}")
        
        # Read CSV header to get actual columns
        with open(csv_file_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            csv_header = next(reader)
        
        logging.info(f"CSV has {len(csv_header)} columns")
        logging.info(f"CSV columns: {csv_header}")
        
        # Match CSV columns to database columns (case-insensitive)
        db_cols_lower = {col.lower(): col for col in column_types.keys()}
        
        logging.info(f"Database columns (lowercase): {sorted(db_cols_lower.keys())}")
        
        columns = []
        csv_to_db_mapping = {}  # Maps CSV index to DB column name
        skipped_columns = []
        
        for idx, csv_col in enumerate(csv_header):
            csv_col_clean = csv_col.strip().lower()
            if csv_col_clean in db_cols_lower:
                db_col_name = db_cols_lower[csv_col_clean]
                columns.append(db_col_name)
                csv_to_db_mapping[idx] = db_col_name
                logging.info(f"  ✓ Matched: '{csv_col}' -> '{db_col_name}'")
            else:
                skipped_columns.append(csv_col)
                logging.warning(f"  ✗ CSV column '{csv_col}' not found in database schema")
        
        if skipped_columns:
            logging.warning(f"Skipped columns: {skipped_columns}")
        
        logging.info(f"Matched {len(columns)} columns for import")
        logging.info(f"{'='*60}\n")
        
        if not columns:
            raise ValueError(f"No matching columns found between CSV and database for table {table_name}")
        
        # Determine columns to update on conflict
        if update_columns is None:
            update_columns = [col for col in columns if col not in conflict_columns]

        # Build UPSERT query with proper type casting
        values_list = []
        for col in columns:
            col_type = column_types.get(col, 'text')
            
            if col_type in ['integer', 'bigint', 'smallint']:
                values_list.append("(NULLIF(%s, '')::NUMERIC)::INTEGER")
            elif col_type in ['numeric', 'decimal']:
                values_list.append("CAST(NULLIF(%s, '') AS NUMERIC)")
            elif col_type in ['real', 'double precision']:
                values_list.append("CAST(NULLIF(%s, '') AS FLOAT)")
            elif col_type == 'date':
                values_list.append("CAST(NULLIF(%s, '') AS DATE)")
            elif col_type == 'time' or col_type == 'time without time zone':
                values_list.append("CAST(NULLIF(%s, '') AS TIME)")
            elif col_type == 'timestamp' or col_type == 'timestamp without time zone':
                values_list.append("CAST(NULLIF(%s, '') AS TIMESTAMP)")
            elif col_type in ['jsonb', 'json']:
                values_list.append("CAST(NULLIF(%s, '') AS JSONB)")
            else:
                values_list.append("NULLIF(%s, '')")
                
        placeholders = ','.join(values_list)
            
        update_set = ','.join([f"{col} = EXCLUDED.{col}" for col in update_columns])

        query = f"""
            INSERT INTO {table_name} ({','.join(columns)})
            VALUES ({placeholders})
            ON CONFLICT ({','.join(conflict_columns)})
            DO UPDATE SET {update_set}
        """
        
        logging.debug(f"Generated SQL:\n{query}\n")

        rows_processed = 0
        batch = []

        try:
            with open(csv_file_path, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                header = next(reader)  # Skip header
                
                for row in reader:
                    # Extract only the columns that matched, in correct order
                    matched_row = []
                    for idx, col_name in csv_to_db_mapping.items():
                        value = row[idx] if idx < len(row) else ''
                        matched_row.append(value)
                    
                    batch.append(matched_row)
                    
                    if len(batch) >= batch_size:
                        execute_batch(cur, query, batch, page_size=100)
                        rows_processed += len(batch)
                        batch = []
                
                # Process remaining rows
                if batch:
                    execute_batch(cur, query, batch, page_size=100)
                    rows_processed += len(batch)
            
            conn.commit()
            logging.info(f"Processed {rows_processed} rows for {table_name}")
            return rows_processed
            
        except Exception as e:
            conn.rollback()
            logging.error(f"Error in {table_name}: {e}")
            raise

def import_hotels_from_local(csv_file_path: str) -> int:
    """Import hotels data from local CSV"""
    return batch_upsert_csv_auto(
        csv_file_path,
        'hotels',
        conflict_columns=['hotel_id']
    )


def import_rooms_from_local(csv_file_path: str) -> int:
    """Import rooms data from local CSV"""
    return batch_upsert_csv_auto(
        csv_file_path,
        'rooms',
        conflict_columns=['hotel_id', 'room_type']
    )


def import_reviews_from_local(csv_file_path: str) -> int:
    """Import reviews data from local CSV"""
    return batch_upsert_csv_auto(
        csv_file_path,
        'reviews',
        conflict_columns=['hotel_id', 'reviewer_name', 'review_date']
    )


def import_amenities_from_local(csv_file_path: str) -> int:
    """Import amenities data from local CSV"""
    return batch_upsert_csv_auto(
        csv_file_path,
        'amenities',
        conflict_columns=['hotel_id', 'category']
    )


def import_policies_from_local(csv_file_path: str) -> int:
    """Import policies data from local CSV"""
    return batch_upsert_csv_auto(
        csv_file_path,
        'policies',
        conflict_columns=['hotel_id']
    )


def load_all_hotel_data_to_database(
    csv_directory: str = LOCAL_CSV_PATH,
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
        ('reviews', 'boston_reviews.csv', import_reviews_from_local),
        ('amenities', 'amenities.csv', import_amenities_from_local),
        ('policies', 'policies.csv', import_policies_from_local)
    ]
    
    for table_name, csv_filename, import_func in import_order:
        csv_path = csv_dir / csv_filename
        
        if not csv_path.exists():
            logging.warning(f"File not found: {csv_path}")
            results[table_name] = 0
            continue
        
        try:
            logging.info(f"\nImporting {table_name}...")
            rows = import_func(str(csv_path))
            results[table_name] = rows
            
        except Exception as e:
            logging.error(f" Failed to import {table_name}: {e}")
            results[table_name] = -1
    
    # Summary
    logging.info("\n" + "="*60)
    logging.info("IMPORT SUMMARY")
    logging.info("="*60)
    
    for table, rows in results.items():
        if rows >= 0:
            logging.info(f"{table}: {rows} rows")
        else:
            logging.info(f" {table}: Failed")
    
    return results