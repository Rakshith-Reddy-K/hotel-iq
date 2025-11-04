# hotel-iq — data_pipeline

This README explains how to set up the environment, run the pipeline, and the code structure for the data_pipeline component.

## Quick overview

- Purpose: Extract hotel metadata and reviews, transform/validate, and load into a Postgres (Cloud) database.
- Stack: Airflow (local via docker-compose), Python scripts (ETL), Postgres (Cloud SQL or local).
- Data flow: Raw data from GCS → Filtered by city → Batched processing → Enrichment via Gemini API → Database load.

## Prerequisites

- Linux machine (project tested on Linux).
- Docker & docker-compose.
- Python 3.9+ and pip.
- Access/credentials for your cloud DB (Cloud SQL or other Postgres).
- Google Cloud SDK if using GCS and Cloud SQL Proxy.
- Google Cloud Storage bucket with raw hotel and review data.

## Setup

1. Clone repo and change to pipeline folder

```bash
cd [your_local_folder]/hotel-iq/data_pipeline
```

2. Install Python dependencies

```bash
python -m pip install -r requirements.txt
```

3. Configure environment

- Create a `.env` file in the `data_pipeline/` directory (same directory as `docker-compose.yaml`). Required variables:
  - `CLOUD_DB_HOST` - Database host
  - `CLOUD_DB_PORT` - Database port
  - `CLOUD_DB_NAME` - Database name
  - `CLOUD_DB_USER` - Database user
  - `CLOUD_DB_PASSWORD` - Database password
  - `GEMINI_API_KEY` - Google Gemini API key for hotel enrichment
  - `GCS_RAW_HOTELS_DATA_PATH` - GCS path to raw hotels data (e.g., `raw/hotels.txt`)
  - `GCS_RAW_REVIEWS_DATA_PATH` - GCS path to raw reviews data (e.g., `raw/reviews.txt`)
  - `GCS_BUCKET_NAME` - Google Cloud Storage bucket name

4. (If using Cloud SQL) Start Cloud SQL Proxy

- Edit `scripts/run_proxy.sh` (project, region, instance name, credentials), then run:

```bash
bash scripts/run_proxy.sh
```

This creates a local socket/port that the database connection pool can connect to, enabling secure access to Cloud DB.

## Run Airflow (local dev)

- **First time setup or after updating `requirements.txt`**: Build and start services (Postgres, Redis, Airflow) using docker-compose:

```bash
docker-compose up -d --build
```

- **Subsequent starts** (if containers already exist): Start services:

```bash
docker-compose up -d
```

- **Note**: If you update `requirements.txt`, you must rebuild the containers to install new packages:
  ```bash
  docker-compose down
  docker-compose up -d --build
  ```

- Airflow web UI: http://localhost:8081 (default credentials: `airflow` / `airflow`)
- The DAG definition is located at `dags/data_pipeline_airflow.py`. Ensure docker-compose mounts the `dags/` directory or copy the DAG into the Airflow DAGs folder.

## How data is processed and uploaded to Cloud DB (summary of process)

The pipeline processes hotel data in batches through the following stages:

1. **Filtering** (`filtering.py`):
   - Downloads raw hotels and reviews from GCS
   - Filters hotels and reviews by city
   - Uploads filtered data back to GCS

2. **Batch Selection** (`batch_selection.py`):
   - Selects next batch of unprocessed hotels based on state tracking
   - Filters reviews for hotels in the current batch
   - Writes batch files locally and to GCS

3. **Enrichment** (`transform.py` + `utils_gemini.py`):
   - Computes aggregate ratings from reviews
   - Enriches hotels with metadata using Gemini Live 2.5 Flash API
   - Merges enrichment data with hotel data

4. **Database Load** (`sql/queries.py`):
   - Uses connection pool from `sql/db_pool.py` to acquire psycopg2 connections
   - Main entrypoint: `bulk_insert_from_csvs(csv_directory)`:
     - Reads processed CSV files (batch_hotels.csv, batch_rooms.csv, batch_amenities.csv, batch_policies.csv, batch_reviews.csv)
     - Groups related data by hotel_id
     - Calls `insert_one_hotel_complete()` for each hotel:
       - Inserts hotel with ON CONFLICT handling
       - Batch inserts rooms, amenities, policies, and reviews using `psycopg2.extras.execute_values`
       - Uses ON CONFLICT DO NOTHING to handle duplicates
     - Returns success/error counts

5. **State Management** (`state_management.py`):
   - Updates processing state with completed batch information
   - Tracks processed hotel IDs to avoid duplicates

6. **Accumulation** (`accumulated.py`):
   - Appends batch results to accumulated files in GCS
   - Maintains city-wide aggregated datasets

## Files and code structure (detailed)

### Root files

- `docker-compose.yaml`
  - Local stack for Airflow (Postgres + Redis + Airflow web/scheduler/worker).

- `requirements.txt`
  - Python dependencies (Airflow, psycopg2, pandas, python-dotenv, google-genai, etc).

### dags/

- `data_pipeline_airflow.py`
  - Airflow DAG wiring for filtering, batching, enrichment (Gemini), aggregation, export, and loading.
  - Defines task dependencies and orchestrates the full pipeline.

- `src/`
  - `transform.py` — Enrichment, aggregation, merge/export helpers. Key functions:
    - `compute_aggregate_ratings(city)` — Computes per-hotel mean ratings from reviews
    - `prepare_hotel_data_for_db(city)` — Merges batch hotels + enrichment + ratings into normalized CSVs
  
  - `filtering.py` — Filter all-city hotels/reviews and per-batch reviews. Key functions:
    - `check_if_filtering_needed(city)` — Checks GCS for previously filtered data; returns branch key
    - `filter_all_city_hotels(city)` — Filters master hotels dataset for a given city; writes city CSV
    - `filter_all_city_reviews(city)` — Filters master reviews dataset for the city; writes reviews CSV
  
  - `batch_selection.py` — Selects next batch of hotel IDs based on state; prepares batch files. Key functions:
    - `select_next_batch(city, batch_size)` — Selects next batch of unprocessed hotels; writes `batch_hotels.csv`
    - `filter_reviews_for_batch(city)` — Filters reviews for hotels in current batch; writes `batch_reviews.csv`
  
  - `accumulated.py` — Appends completed batch outputs to accumulated results in GCS. Key functions:
    - `append_batch_to_accumulated(city)` — Appends batch results to accumulated artifacts in GCS
  
  - `state_management.py` — Maintains `state.json` per city; tracks processed IDs and batches. Key functions:
    - `update_processing_state(city)` — Updates state.json with processed IDs and batch metadata
  
  - `utils.py` — Parsing, cleaning, enrichment merge utilities. Key functions:
    - `parse_raw_hotels(file_path)` — Parses JSONL hotel data into DataFrame
    - `parse_raw_reviews(file_path)` — Parses JSONL review data into DataFrame
    - `calculate_all_hotel_ratings(reviews_df)` — Computes aggregate ratings per hotel
    - `merge_hotel_data(row, enrichment_data)` — Merges hotel row with Gemini enrichment data
    - `export_hotel_data_to_csv(data_dict, output_dir)` — Exports normalized hotel data to CSVs
  
  - `utils_gemini.py` — Gemini Live 2.5 Flash client and prompt formatting. Key functions:
    - `get_hotel_data(hotel_name, location)` — Synchronous wrapper for async Gemini API call
    - `get_hotel_data_async(hotel_name, location)` — Async function to get hotel data from Gemini API
    - `create_prompt(hotel_name, location)` — Creates structured prompt for hotel data extraction
  
  - `prompt.py` — Hotel data extraction functions (legacy Perplexity support + Gemini wrapper). Key functions:
    - `extract_hotel_data_gemini(hotel_name, location)` — Extracts hotel data using Gemini (delegates to utils_gemini)
    - `extract_hotel_data_perplexity(hotel_name, location)` — Legacy Perplexity API extraction
    - `enrich_hotels_gemini(city, delay_seconds, max_hotels)` — Enriches hotels in batch using Gemini
  
  - `path.py` — Path management utilities for local and GCS paths. Key functions:
    - `get_raw_hotels_path()` — Returns path to raw hotels file
    - `get_filtered_hotels_path(city)` — Returns path to filtered hotels CSV for city
    - `get_batch_hotels_path(city)` — Returns path to batch hotels CSV
    - `get_processed_dir(city)` — Returns path to processed data directory
    - `get_gcs_*_path()` functions — Returns GCS paths for various data types
  
  - `bucket_util.py` — Helpers for reading/writing Google Cloud Storage. Key functions:
    - `upload_file_to_gcs(local_path, gcs_path)` — Uploads file to GCS
    - `download_file_from_gcs(gcs_path, local_path)` — Downloads file from GCS

### scripts/

- `run_proxy.sh` — Helper to start Cloud SQL Proxy; edit for your project/credentials.

### sql/

- `db_pool.py`
  - Implements connection pool singleton using `psycopg2.pool.ThreadedConnectionPool`
  - Exposes `get_connection()` context manager for database connections
  - Reads DB connection parameters from environment variables (CLOUD_DB_*)
  - Initializes pool on module import

- `queries.py`
  - SQL DDL helpers to create tables and insert data. Key functions:
    - `create_all_tables()` — Creates all required tables (hotels, rooms, reviews, amenities, policies)
    - `list_tables()` — Lists all tables in the database
    - `bulk_insert_from_csvs(csv_directory)` — Main entrypoint: reads processed CSVs and loads into database
    - `insert_one_hotel_complete(hotel_data, rooms, amenities, policies, reviews)` — Inserts one hotel with all related data
    - `clean_dict_for_db(d)` — Converts NaN values to None for database compatibility

- `test_connection.py`
  - Simple script to validate DB connectivity and list tables.

## CSV format expectations

The pipeline expects processed CSV files in the following format:

- `batch_hotels.csv` — Hotel records with columns matching the hotels table schema
- `batch_rooms.csv` — Room records with `hotel_id` foreign key
- `batch_amenities.csv` — Amenity records with `hotel_id` foreign key
- `batch_policies.csv` — Policy records with `hotel_id` foreign key
- `batch_reviews.csv` — Review records with `hotel_id` foreign key

Each CSV must include a header row with column names matching the database table schemas defined in `sql/queries.py`. Missing cells should be empty strings or NaN; the loader converts NaN values to NULL.

## Database schema

The database consists of five main tables:

1. **hotels** — Main hotel information (hotel_id, official_name, star_rating, description, address, ratings, etc.)
2. **rooms** — Room types for each hotel (room_id, hotel_id, room_type, bed_configuration, etc.)
3. **reviews** — Reviews for hotels (review_id, hotel_id, overall_rating, review_text, etc.)
4. **amenities** — Amenities by category (amenity_id, hotel_id, category, description, etc.)
5. **policies** — Hotel policies (policy_id, hotel_id, check_in_time, check_out_time, etc.)

See `sql/queries.py` for complete table definitions with constraints and data types.

## Tips & troubleshooting

- If you use Cloud SQL, confirm Cloud SQL Proxy is running and that the connection pool connects to the proxy host/port.
- Check logs for stack traces; all modules use Python logging.
- Validate table schemas (types and primary keys) to ensure ON CONFLICT keys align with CSV columns.
- For large loads, the pipeline processes hotels in batches (default 50) to avoid memory issues.
- Ensure GCS bucket permissions are correctly configured for read/write operations.
- Check that `GEMINI_API_KEY` is set correctly if enrichment tasks are failing.

## Example — run full pipeline

1. Start Cloud SQL Proxy (if required).
2. Ensure `.env` is present with all required connection details and API keys.
3. Build and start Airflow (this installs requirements from `requirements.txt`): `docker-compose up -d --build`
4. Access Airflow UI: http://localhost:8081 (credentials: `airflow` / `airflow`)
5. Trigger the DAG for your city (e.g., "Boston")
6. Monitor progress in the Airflow UI

The pipeline will:
- Filter raw data by city
- Process hotels in batches
- Enrich with Gemini API
- Load into database
- Update state and accumulate results
