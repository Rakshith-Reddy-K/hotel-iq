import os
import time
import json
import logging
from pathlib import Path
import shutil
import pandas as pd

from src.utils import get_reviews_for_hotels, calculate_all_hotel_ratings, extract_hotel_data_from_row, merge_hotel_data, export_hotel_data_to_csv

# Configure logging
logger = logging.getLogger(__name__)


def _ensure_output_dir(output_dir: str) -> None:
    Path(output_dir).mkdir(parents=True, exist_ok=True)


def _resolve_project_path(path_str: str) -> str:
    path = Path(path_str)
    if path.is_absolute():
        return str(path)
    project_root = Path(__file__).resolve().parents[2]
    return str((project_root / path).resolve())


def extract_reviews_based_on_city(
    city: str = 'Boston',
    output_dir: str = 'output'
):
    """
    Filter reviews for specific city hotels.
    
    Returns:
        str: Path to filtered reviews CSV
    """
    logger.info(f"Starting city-specific review filtering for: {city}")
    
    try:
        city_token = city.title().replace(' ', '')
        city_lower = city.lower().replace(' ', '_')

        output_abspath = _resolve_project_path(output_dir)
        _ensure_output_dir(output_abspath)

        hotel_csv_path = os.path.join(output_abspath, f'hotel_data_{city_token}.csv')
        reviews_csv_path = os.path.join(output_abspath, 'reviews.csv')
        output_path = os.path.join(output_abspath, f'{city_lower}_reviews.csv')

        logger.info(f"Reading hotel data from: {hotel_csv_path}")
        logger.info(f"Reading reviews data from: {reviews_csv_path}")
        
        filtered_reviews = get_reviews_for_hotels(hotel_csv_path, reviews_csv_path, output_path)
        logger.info(f"Successfully filtered {len(filtered_reviews)} reviews for {city}")
        logger.info(f"Filtered reviews saved to: {output_path}")

        return str(output_path)
        
    except Exception as e:
        logger.error(f"Failed to filter city reviews: {str(e)}")
        raise


def compute_aggregate_ratings(
    city: str = 'Boston',
):
    city_lower = city.lower().replace(' ', '_')

    output_abspath = _resolve_project_path(f'data/processing/{city_lower}')
    _ensure_output_dir(output_abspath)

    city_reviews_path = os.path.join(output_abspath, 'batch_reviews.csv')
    if os.path.exists(city_reviews_path):
        reviews_df = pd.read_csv(city_reviews_path)
    else:
        reviews_df = pd.read_csv(os.path.join(f'data/filtered/{city_lower}', 'reviews.csv'))

    agg_df = calculate_all_hotel_ratings(reviews_df)
    out_path = os.path.join(output_abspath, 'batch_hotel_ratings.csv')
    agg_df.to_csv(out_path, index=False)
    return out_path


def enrich_hotels_perplexity(
    city: str = 'Boston',
    delay_seconds: float = 12,
    max_hotels: int = None
):
    city_lower = city.lower().replace(' ', '_')
    output_abspath = _resolve_project_path(f'data/processing/{city_lower}')
    _ensure_output_dir(output_abspath)

    hotels_csv = os.path.join(output_abspath, 'batch_hotels.csv')
    if not os.path.exists(hotels_csv):
        raise FileNotFoundError(f"Expected hotels CSV not found: {hotels_csv}")

    df = pd.read_csv(hotels_csv)
    if max_hotels is not None:
        df = df.head(max_hotels)

    enrichment_path = os.path.join(output_abspath, 'batch_enrichment.jsonl')

    with open(enrichment_path, 'w', encoding='utf-8') as f:
        for _, row in df.iterrows():
            try:
                data = extract_hotel_data_from_row(row)
                record = {
                    'hotel_id': row.get('id'),
                    'name': row.get('name'),
                    'city': row.get('address_locality'),
                    'data': data
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
            except Exception as e:
                err_record = {
                    'hotel_id': row.get('id'),
                    'name': row.get('name'),
                    'error': str(e)
                }
                f.write(json.dumps(err_record, ensure_ascii=False) + "\n")
            time.sleep(delay_seconds)

    return enrichment_path


def prepare_hotel_data_for_db(
    city: str = 'Boston',
):
    city_lower = city.lower().replace(' ', '_')

    output_abspath = _resolve_project_path(f'data/processing/{city_lower}')
    _ensure_output_dir(output_abspath)

    hotels_csv = os.path.join(output_abspath, 'batch_hotels.csv')
    enrichment_path = os.path.join(output_abspath, 'batch_enrichment.jsonl')
    ratings_csv = os.path.join(output_abspath, 'batch_hotel_ratings.csv')
    reviews_csv = os.path.join(output_abspath, 'batch_reviews.csv')

    if not os.path.exists(hotels_csv):
        raise FileNotFoundError(f"Expected hotels CSV not found: {hotels_csv}")
    if not os.path.exists(enrichment_path):
        raise FileNotFoundError(f"Expected enrichment JSONL not found: {enrichment_path}")

    hotels_df = pd.read_csv(hotels_csv)

    # Load enrichment into dict by hotel_id
    hotel_id_to_data = {}
    with open(enrichment_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                hid = obj.get('hotel_id')
                data = obj.get('data')
                # Only keep entries that have a non-null 'hotel' payload
                if hid is not None and isinstance(data, dict) and data.get('hotel') is not None:
                    hotel_id_to_data[hid] = data
            except Exception:
                continue

    # Accumulate tables
    hotels_acc = []
    rooms_acc = []
    amenities_acc = []
    policies_acc = []

    for _, row in hotels_df.iterrows():
        hid = row.get('id')
        merged = merge_hotel_data(row, hotel_id_to_data.get(hid))
        if not merged['hotels'].empty:
            hotels_acc.append(merged['hotels'])
        if not merged['rooms'].empty:
            rooms_acc.append(merged['rooms'])
        if not merged['amenities'].empty:
            amenities_acc.append(merged['amenities'])
        if not merged['policies'].empty:
            policies_acc.append(merged['policies'])

    hotels_out = pd.concat(hotels_acc, ignore_index=True) if hotels_acc else pd.DataFrame()
    rooms_out = pd.concat(rooms_acc, ignore_index=True) if rooms_acc else pd.DataFrame()
    amenities_out = pd.concat(amenities_acc, ignore_index=True) if amenities_acc else pd.DataFrame()
    policies_out = pd.concat(policies_acc, ignore_index=True) if policies_acc else pd.DataFrame()

    # Join rating aggregates if present
    if os.path.exists(ratings_csv) and not hotels_out.empty:
        ratings_df = pd.read_csv(ratings_csv)
        rating_cols = ['overall_rating', 'total_reviews', 'cleanliness_rating', 'service_rating', 'location_rating', 'value_rating']
        hotels_out = hotels_out.drop(columns=[col for col in rating_cols if col in hotels_out.columns])
        hotels_out = hotels_out.merge(ratings_df, how='left', on='hotel_id')

    
    processed_folder = _resolve_project_path(f'data/processed/{city_lower}')
    _ensure_output_dir(processed_folder)
    export_hotel_data_to_csv(
        {
            'hotels': hotels_out,
            'rooms': rooms_out,
            'amenities': amenities_out,
            'policies': policies_out,
        },
        output_dir=processed_folder
    )
    if os.path.exists(enrichment_path):
        enrichment_dest = os.path.join(processed_folder, 'batch_enrichment.jsonl')
        shutil.copy(enrichment_path, enrichment_dest)

    if os.path.exists(reviews_csv):
            reviews_dest = os.path.join(processed_folder, 'batch_reviews.csv')
            shutil.copy(reviews_csv, reviews_dest)

    return {
        'hotels': os.path.join(processed_folder, 'batch_hotels.csv'),
        'rooms': os.path.join(processed_folder, 'batch_rooms.csv'),
        'amenities': os.path.join(processed_folder, 'batch_amenities.csv'),
        'policies': os.path.join(processed_folder, 'batch_policies.csv'),
        'reviews': os.path.join(processed_folder, 'batch_reviews.csv'),
        'enrichment': enrichment_dest
    }