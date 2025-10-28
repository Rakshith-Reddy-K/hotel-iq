import os
import logging
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple

# Configure logging
logger = logging.getLogger(__name__)


def _resolve_project_path(path_str: str) -> str:
    path = Path(path_str)
    if path.is_absolute():
        return str(path)
    project_root = Path(__file__).resolve().parents[2]
    return str((project_root / path).resolve())


def validate_data(
    city: str = 'Boston',
    output_dir: str = 'output'
) -> Dict[str, Dict]:
    """
    Validate SQL-ready datasets for schema compliance, data quality, and completeness.
    
    Returns:
        Dict with validation results for each table
    """
    logger.info(f"Starting data validation for city: {city}")
    
    city_token = city.title().replace(' ', '')
    city_lower = city.lower().replace(' ', '_')
    
    output_abspath = _resolve_project_path(output_dir)
    
    validation_results = {}
    
    # Define expected schemas
    schemas = {
        'hotels': [
            'hotel_id', 'official_name', 'star_rating', 'description', 'address',
            'city', 'state', 'zip_code', 'country', 'phone', 'email', 'website',
            'overall_rating', 'total_reviews', 'cleanliness_rating', 'service_rating',
            'location_rating', 'value_rating', 'year_opened', 'last_renovation',
            'total_rooms', 'number_of_floors', 'additional_info'
        ],
        'rooms': [
            'hotel_id', 'room_type', 'bed_configuration', 'room_size_sqft',
            'max_occupancy', 'price_range_min', 'price_range_max', 'description'
        ],
        'amenities': [
            'hotel_id', 'category', 'description', 'details'
        ],
        'policies': [
            'hotel_id', 'check_in_time', 'check_out_time', 'min_age_requirement',
            'pet_policy', 'smoking_policy', 'children_policy', 'extra_person_policy',
            'cancellation_policy'
        ],
        'reviews': [
            'review_id', 'hotel_id', 'overall_rating', 'review_text',
            'reviewer_name', 'review_date', 'source', 'service_rating',
            'cleanliness_rating', 'value_rating', 'location_rating'
        ]
    }
    
    # Validate each table
    tables = ['hotels', 'rooms', 'amenities', 'policies']
    
    for table in tables:
        csv_path = os.path.join(output_abspath, f'{table}.csv')
        
        if not os.path.exists(csv_path):
            validation_results[table] = {
                'status': 'MISSING',
                'message': f'File not found: {csv_path}',
                'row_count': 0,
                'schema_valid': False
            }
            continue
            
        try:
            df = pd.read_csv(csv_path)
            expected_cols = schemas.get(table, [])
            
            # Schema validation
            missing_cols = set(expected_cols) - set(df.columns)
            extra_cols = set(df.columns) - set(expected_cols)
            schema_valid = len(missing_cols) == 0
            
            # Data quality checks
            null_counts = df.isnull().sum().to_dict()
            total_nulls = df.isnull().sum().sum()
            
            # Filter to only show columns with nulls
            columns_with_nulls = {col: count for col, count in null_counts.items() if count > 0}
            
            # Range validations for numeric fields
            range_violations = []
            if table == 'hotels':
                if 'star_rating' in df.columns:
                    invalid_stars = df[~df['star_rating'].between(1, 5, na=True)]
                    if len(invalid_stars) > 0:
                        range_violations.append(f"star_rating: {len(invalid_stars)} values outside 1-5 range")
                        
            elif table == 'rooms':
                if 'room_size_sqft' in df.columns:
                    invalid_size = df[df['room_size_sqft'] < 0]
                    if len(invalid_size) > 0:
                        range_violations.append(f"room_size_sqft: {len(invalid_size)} negative values")
                        
            elif table == 'reviews':
                if 'overall_rating' in df.columns:
                    invalid_rating = df[~df['overall_rating'].between(1, 5, na=True)]
                    if len(invalid_rating) > 0:
                        range_violations.append(f"overall_rating: {len(invalid_rating)} values outside 1-5 range")
            
            validation_results[table] = {
                'status': 'VALID' if schema_valid and len(range_violations) == 0 else 'ISSUES',
                'row_count': len(df),
                'schema_valid': schema_valid,
                'missing_columns': list(missing_cols),
                'extra_columns': list(extra_cols),
                'null_counts': columns_with_nulls,  # Only show columns with nulls
                'total_nulls': int(total_nulls),
                'range_violations': range_violations,
                'message': f"Table {table}: {len(df)} rows, schema valid: {schema_valid}"
            }
            
        except Exception as e:
            validation_results[table] = {
                'status': 'ERROR',
                'message': f'Error reading {csv_path}: {str(e)}',
                'row_count': 0,
                'schema_valid': False
            }
    
    # Validate city-specific reviews
    city_reviews_path = os.path.join(output_abspath, f'{city_lower}_reviews.csv')
    if os.path.exists(city_reviews_path):
        try:
            reviews_df = pd.read_csv(city_reviews_path)
            expected_cols = schemas.get('reviews', [])
            
            missing_cols = set(expected_cols) - set(reviews_df.columns)
            schema_valid = len(missing_cols) == 0
            
            validation_results['city_reviews'] = {
                'status': 'VALID' if schema_valid else 'ISSUES',
                'row_count': len(reviews_df),
                'schema_valid': schema_valid,
                'missing_columns': list(missing_cols),
                'message': f"City reviews: {len(reviews_df)} rows, schema valid: {schema_valid}"
            }
        except Exception as e:
            validation_results['city_reviews'] = {
                'status': 'ERROR',
                'message': f'Error reading city reviews: {str(e)}',
                'row_count': 0,
                'schema_valid': False
            }
    
    # Summary
    total_issues = sum(1 for result in validation_results.values() if result['status'] != 'VALID')
    print(f"\n{'='*60}")
    print(f"VALIDATION SUMMARY")
    print(f"{'='*60}")
    print(f"Total tables checked: {len(validation_results)}")
    print(f"Tables with issues: {total_issues}")
    
    for table, result in validation_results.items():
        status_icon = "✅" if result['status'] == 'VALID' else "❌" if result['status'] == 'ERROR' else "⚠️"
        print(f"{status_icon} {table}: {result['message']}")
        if result['status'] != 'VALID' and 'missing_columns' in result:
            if result['missing_columns']:
                print(f"   Missing columns: {result['missing_columns']}")
        if result['status'] != 'VALID' and 'range_violations' in result:
            if result['range_violations']:
                print(f"   Range violations: {result['range_violations']}")
        # Always show null counts if any exist
        if 'null_counts' in result and result['null_counts']:
            print(f"   Columns with nulls: {result['null_counts']}")
    
    return validation_results