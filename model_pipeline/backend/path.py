from pathlib import Path
from typing import Optional

BASE_DATA_DIR = "data"
RAW_DIR = f"{BASE_DATA_DIR}/raw"
FILTERED_DIR = f"{BASE_DATA_DIR}/filtered"
PROCESSING_DIR = f"{BASE_DATA_DIR}/processing"
PROCESSED_DIR = f"{BASE_DATA_DIR}/processed"

RAW_HOTELS_FILENAME = "hotels.txt"
RAW_REVIEWS_FILENAME = "reviews.txt"

BATCH_HOTELS_FILENAME = "batch_hotels.csv"
BATCH_REVIEWS_FILENAME = "batch_reviews.csv"
BATCH_ENRICHMENT_FILENAME = "batch_enrichment.jsonl"
BATCH_RATINGS_FILENAME = "batch_hotel_ratings.csv"

HOTELS_FILENAME = "hotels.csv"
REVIEWS_FILENAME = "reviews.csv"
ROOMS_FILENAME = "rooms.csv"
AMENITIES_FILENAME = "amenities.csv"
POLICIES_FILENAME = "policies.csv"
ENRICHMENT_FILENAME = "enrichment.jsonl"
STATE_FILENAME = "state.json"

TABLE_NAMES = ['hotels', 'rooms', 'amenities', 'policies', 'reviews']

GCS_RAW_PREFIX = "raw"
GCS_FILTERED_PREFIX = "filtered"
GCS_PROCESSED_PREFIX = "processed"

def _resolve_and_ensure(path_str: str, ensure_dir: bool = True) -> str:
    path = Path(path_str)
    if path.is_absolute():
        if ensure_dir:
            path.parent.mkdir(parents=True, exist_ok=True)
        return str(path)
    # Go up 1 level from backend/path.py to get to project root
    project_root = Path(__file__).resolve().parents[1]
    resolved_path = (project_root / path).resolve()
    if ensure_dir:
        resolved_path.parent.mkdir(parents=True, exist_ok=True)
    return str(resolved_path)


def get_city_folder(city: str) -> str:
    return city.lower().replace(' ', '_')


def get_raw_hotels_path() -> str:
    return _resolve_and_ensure(f"{RAW_DIR}/{RAW_HOTELS_FILENAME}")


def get_raw_reviews_path() -> str:
    return _resolve_and_ensure(f"{RAW_DIR}/{RAW_REVIEWS_FILENAME}")


def get_filtered_dir(city: str) -> str:
    city_folder = get_city_folder(city)
    path = _resolve_and_ensure(f"{FILTERED_DIR}/{city_folder}", ensure_dir=False)
    Path(path).mkdir(parents=True, exist_ok=True)
    return path


def get_filtered_hotels_path(city: str) -> str:
    city_folder = get_city_folder(city)
    return _resolve_and_ensure(f"{FILTERED_DIR}/{city_folder}/{HOTELS_FILENAME}")


def get_filtered_reviews_path(city: str) -> str:
    city_folder = get_city_folder(city)
    return _resolve_and_ensure(f"{FILTERED_DIR}/{city_folder}/{REVIEWS_FILENAME}")


def get_processing_dir(city: str) -> str:
    city_folder = get_city_folder(city)
    path = _resolve_and_ensure(f"{PROCESSING_DIR}/{city_folder}", ensure_dir=False)
    Path(path).mkdir(parents=True, exist_ok=True)
    return path


def get_batch_hotels_path(city: str) -> str:
    city_folder = get_city_folder(city)
    return _resolve_and_ensure(f"{PROCESSING_DIR}/{city_folder}/{BATCH_HOTELS_FILENAME}")


def get_batch_reviews_path(city: str) -> str:
    city_folder = get_city_folder(city)
    return _resolve_and_ensure(f"{PROCESSING_DIR}/{city_folder}/{BATCH_REVIEWS_FILENAME}")


def get_batch_enrichment_path(city: str) -> str:
    city_folder = get_city_folder(city)
    return _resolve_and_ensure(f"{PROCESSING_DIR}/{city_folder}/{BATCH_ENRICHMENT_FILENAME}")


def get_batch_ratings_path(city: str) -> str:
    city_folder = get_city_folder(city)
    return _resolve_and_ensure(f"{PROCESSING_DIR}/{city_folder}/{BATCH_RATINGS_FILENAME}")


def get_processed_dir(city: str) -> str:
    city_folder = get_city_folder(city)
    path = _resolve_and_ensure(f"{PROCESSED_DIR}/{city_folder}", ensure_dir=False)
    Path(path).mkdir(parents=True, exist_ok=True)
    return path


def get_processed_hotels_path(city: str) -> str:
    city_folder = get_city_folder(city)
    return _resolve_and_ensure(f"{PROCESSED_DIR}/{city_folder}/{HOTELS_FILENAME}")


def get_processed_reviews_path(city: str) -> str:
    city_folder = get_city_folder(city)
    return _resolve_and_ensure(f"{PROCESSED_DIR}/{city_folder}/{REVIEWS_FILENAME}")


def get_processed_rooms_path(city: str) -> str:
    city_folder = get_city_folder(city)
    return _resolve_and_ensure(f"{PROCESSED_DIR}/{city_folder}/{ROOMS_FILENAME}")


def get_processed_amenities_path(city: str) -> str:
    city_folder = get_city_folder(city)
    return _resolve_and_ensure(f"{PROCESSED_DIR}/{city_folder}/{AMENITIES_FILENAME}")


def get_processed_policies_path(city: str) -> str:
    city_folder = get_city_folder(city)
    return _resolve_and_ensure(f"{PROCESSED_DIR}/{city_folder}/{POLICIES_FILENAME}")


def get_processed_enrichment_path(city: str) -> str:
    city_folder = get_city_folder(city)
    return _resolve_and_ensure(f"{PROCESSED_DIR}/{city_folder}/{ENRICHMENT_FILENAME}")


def get_processed_state_path(city: str) -> str:
    city_folder = get_city_folder(city)
    return _resolve_and_ensure(f"{PROCESSED_DIR}/{city_folder}/{STATE_FILENAME}")


def get_processed_batch_file_path(city: str, table: str) -> str:
    city_folder = get_city_folder(city)
    return _resolve_and_ensure(f"{PROCESSED_DIR}/{city_folder}/batch_{table}.csv")


def get_processed_table_file_path(city: str, table: str) -> str:
    city_folder = get_city_folder(city)
    return _resolve_and_ensure(f"{PROCESSED_DIR}/{city_folder}/{table}.csv")

def get_gcs_raw_hotels_path() -> str:
    return f"{GCS_RAW_PREFIX}/{RAW_HOTELS_FILENAME}"


def get_gcs_raw_reviews_path() -> str:
    return f"{GCS_RAW_PREFIX}/{RAW_REVIEWS_FILENAME}"


def get_gcs_filtered_hotels_path(city: str) -> str:
    city_folder = get_city_folder(city)
    return f"{GCS_FILTERED_PREFIX}/{city_folder}/{HOTELS_FILENAME}"


def get_gcs_filtered_reviews_path(city: str) -> str:
    city_folder = get_city_folder(city)
    return f"{GCS_FILTERED_PREFIX}/{city_folder}/{REVIEWS_FILENAME}"


def get_gcs_processed_batch_hotels_path(city: str) -> str:
    city_folder = get_city_folder(city)
    return f"{GCS_PROCESSED_PREFIX}/{city_folder}/{BATCH_HOTELS_FILENAME}"


def get_gcs_processed_batch_reviews_path(city: str) -> str:
    city_folder = get_city_folder(city)
    return f"{GCS_PROCESSED_PREFIX}/{city_folder}/{BATCH_REVIEWS_FILENAME}"


def get_gcs_processed_table_path(city: str, table: str) -> str:
    city_folder = get_city_folder(city)
    return f"{GCS_PROCESSED_PREFIX}/{city_folder}/{table}.csv"


def get_gcs_processed_enrichment_path(city: str) -> str:
    city_folder = get_city_folder(city)
    return f"{GCS_PROCESSED_PREFIX}/{city_folder}/{ENRICHMENT_FILENAME}"


def get_gcs_processed_state_path(city: str) -> str:
    city_folder = get_city_folder(city)
    return f"{GCS_PROCESSED_PREFIX}/{city_folder}/{STATE_FILENAME}"
