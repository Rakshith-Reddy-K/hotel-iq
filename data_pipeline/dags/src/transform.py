import os
import json
import logging
import shutil
import pandas as pd

from src.utils import calculate_all_hotel_ratings, merge_hotel_data, export_hotel_data_to_csv
from src.path import (
    get_batch_hotels_path,
    get_batch_reviews_path,
    get_batch_enrichment_path,
    get_batch_ratings_path,
    get_processed_dir,
    get_filtered_reviews_path
)

logger = logging.getLogger(__name__)


def compute_aggregate_ratings(city: str = 'Boston'):
    city_reviews_path = get_batch_reviews_path(city)
    if os.path.exists(city_reviews_path):
        reviews_df = pd.read_csv(city_reviews_path)
    else:
        reviews_df = pd.read_csv(get_filtered_reviews_path(city))

    agg_df = calculate_all_hotel_ratings(reviews_df)
    out_path = get_batch_ratings_path(city)
    agg_df.to_csv(out_path, index=False)
    return out_path

def prepare_hotel_data_for_db(city: str = 'Boston'):
    hotels_csv = get_batch_hotels_path(city)
    enrichment_path = get_batch_enrichment_path(city)
    ratings_csv = get_batch_ratings_path(city)
    reviews_csv = get_batch_reviews_path(city)

    if not os.path.exists(hotels_csv):
        raise FileNotFoundError(f"Expected hotels CSV not found: {hotels_csv}")
    if not os.path.exists(enrichment_path):
        raise FileNotFoundError(f"Expected enrichment JSONL not found: {enrichment_path}")

    hotels_df = pd.read_csv(hotels_csv)

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
                if hid is not None and isinstance(data, dict) and data.get('hotel') is not None:
                    hotel_id_to_data[hid] = data
            except Exception:
                continue

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

    if os.path.exists(ratings_csv) and not hotels_out.empty:
        ratings_df = pd.read_csv(ratings_csv)
        rating_cols = ['overall_rating', 'total_reviews', 'cleanliness_rating', 'service_rating', 'location_rating', 'value_rating']
        hotels_out = hotels_out.drop(columns=[col for col in rating_cols if col in hotels_out.columns])
        hotels_out = hotels_out.merge(ratings_df, how='left', on='hotel_id')

    processed_folder = get_processed_dir(city)
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