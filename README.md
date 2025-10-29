# hotel-iq — data_pipeline

This README explains how to set up the environment, run the pipeline, and the code structure for the data_pipeline component.

## Quick overview

- Purpose: Extract hotel metadata and reviews, transform/validate, and load into a Postgres (Cloud) database.
- Stack: Airflow (local via docker-compose), Python scripts (ETL), Postgres (Cloud SQL or local).
- CSV source: Local CSV files (default directory `intermediate/csv`) → uploaded to DB via the loader.

## Prerequisites

- Linux machine (project tested on Linux).
- Docker & docker-compose.
- Python 3.9+ and pip.
- Access/credentials for your cloud DB (Cloud SQL or other Postgres).
- (Optional) Google Cloud SDK if using Cloud SQL Proxy.

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

- Create a `.env` file in this folder (or set env vars in your shell). Typical variables used by the DB pool:
  - DB_HOST
  - DB_PORT
  - DB_NAME
  - DB_USER
  - DB_PASSWORD
- The loader reads `LOCAL_CSV_PATH` from env (defaults to `intermediate/csv`).

4. (If using Cloud SQL) Start Cloud SQL Proxy

- Edit `scripts/run_proxy.sh` (project, region, instance name, credentials), then run:

```bash
bash scripts/run_proxy.sh
```

This creates a local socket/port that the `db_pool` can connect to, enabling secure access to Cloud DB.

## Run Airflow (local dev)

- Start services (Postgres, Redis, Airflow) using docker-compose:

```bash
docker-compose up -d
```

- Airflow web UI: http://localhost:8080
- The DAG definition is located at `dags/data_pipeline_airflow.py`. Ensure docker-compose mounts the `dags/` directory or copy the DAG into the Airflow DAGs folder.

## Run the loader (upload CSV → DB)

- Verify CSVs exist in the CSV directory (default `intermediate/csv`):

  - hotels.csv, rooms.csv, reviews.csv, amenities.csv, policies.csv

- Run the end-to-end loader from the project root:

```bash
python -c "from sql.load_to_database import load_all_hotel_data_to_database; print(load_all_hotel_data_to_database('intermediate/csv'))"
```

- Or run interactively / as part of a script:

```python
from sql.load_to_database import load_all_hotel_data_to_database
results = load_all_hotel_data_to_database('/path/to/csv_dir')
print(results)
```

- You can also run `sql/test_connection.py` to validate DB connectivity and list tables.

## How local data is uploaded to Cloud DB (summary of process)

- The loader uses the connection pool in `sql/db_pool.py` (singleton `db_pool`) to acquire a psycopg2 connection.
- Main entrypoint: `load_all_hotel_data_to_database(csv_directory)`:

  - Validates `csv_directory` exists.
  - Imports tables in a safe order to honor FK constraints:
    - hotels → rooms → reviews → amenities → policies
  - For each table it calls a table-specific importer:
    - import_hotels_from_local(csv_path)
    - import_rooms_from_local(csv_path)
    - import_reviews_from_local(csv_path)
    - import_amenities_from_local(csv_path)
    - import_policies_from_local(csv_path)

- Table importers call `batch_upsert_csv_auto(...)` which:

  - Opens a DB connection via `get_db_connection()` (wraps `db_pool.get_connection()`).
  - Reads the target table column types from `information_schema.columns` to decide casting rules.
  - Builds an INSERT ... ON CONFLICT ... DO UPDATE query (UPSERT) with proper casts:
    - Numeric/text/date/timestamp columns are cast using `CAST(NULLIF(%s, '') AS <TYPE>)` so empty strings become NULL.
  - Batches rows (default batch_size 1000) and executes them via `psycopg2.extras.execute_batch` for performance.
  - Commits at completion; rolls back on error and raises.

- There is also `batch_upsert_csv(...)` (simpler variant) that directly inserts without automatic type detection/casting — useful when CSV already matches DB types.

- Logging is emitted for progress and summary; returned values are row counts per table, or -1 on failure.

## Files and code structure (detailed)

- docker-compose.yaml

  - Local stack for Airflow (Postgres + Redis + Airflow web/scheduler/worker).

- requirements.txt

  - Python dependencies (Airflow, psycopg2, pandas, python-dotenv, etc).

- dags/

  - data_pipeline_airflow.py
    - Airflow DAG wiring extraction, transform, validation, and loader tasks.
  - src/
    - extract.py — functions to download and save raw hotel metadata and reviews (source files).
    - transform.py — lightweight preprocessing helpers (e.g., review cleanup, city filtering).
    - validation.py — simple data checks and assertions before load.
    - bucket_util.py — optional helpers for reading/writing cloud storage buckets.

Transform helpers (example functions from transform module)

- extract_reviews_based_on_city(city, output_dir)

  - Filters the main reviews CSV to keep only reviews for hotels listed in the city-specific hotel CSV. Writes a city-specific reviews CSV and returns its path.

- compute_aggregate_ratings(city, output_dir)

  - Loads city reviews (or global reviews) and computes per-hotel aggregate ratings (overall, cleanliness, service, location, value). Writes hotel*ratings*{City}.csv.

- enrich_hotels_perplexity(city, output_dir, delay_seconds, max_hotels)

  - Iterates hotels, builds enrichment payloads (via extract_hotel_data_from_row), writes JSONL lines (one per hotel) used for downstream enrichment/ML/API calls. Throttles via delay_seconds.

- merge_sql_tables(city, output_dir)

  - Reads hotels CSV + enrichment JSONL (and ratings CSV if present), merges enrichment into normalized DataFrames (hotels, rooms, amenities, policies), and exports final CSVs ready for DB bulk load.

- scripts/

  - run_proxy.sh — helper to start Cloud SQL Proxy; edit for your project/credentials.

- sql/
  - db_pool.py
    - Implements `DatabasePool` singleton and exposes `db_pool` used by loaders. Reads DB connection parameters from env.
  - load_to_database.py
    - get_db_connection() — returns a psycopg2 connection via db_pool.
    - batch_upsert_csv(...) — batch UPSERT using raw placeholders (no automatic casting).
    - batch_upsert_csv_auto(...) — batch UPSERT with automatic type detection and value casting (preferred).
    - import_hotels_from_local(...) — importer for hotels.csv.
    - import_rooms_from_local(...) — importer for rooms.csv.
    - import_reviews_from_local(...) — importer for reviews.csv.
    - import_amenities_from_local(...) — importer for amenities.csv.
    - import_policies_from_local(...) — importer for policies.csv.
    - load_all_hotel_data_to_database(...) — orchestrator that runs the importers in order and returns a results dict.
  - queries.py
    - SQL DDL helpers to create tables and insert helper snippets. Useful to inspect schema expectations (primary keys, column types) which must match CSV headers listed in load_to_database.py.
  - test_connection.py
    - Simple script to validate DB connectivity and list tables.

## CSV format expectations

- Each CSV must include a header row with column names matching the `columns` lists defined in the import\_\* functions.
- Missing cells should be empty strings; loader casts empty strings to NULL for typed columns.
- File names expected by default:
  - hotels.csv, rooms.csv, reviews.csv, amenities.csv, policies.csv
- If your schema differs adjust the `columns` arrays in `sql/load_to_database.py` to match.

## Tips & troubleshooting

- If you use Cloud SQL, confirm Cloud SQL Proxy is running and that `db_pool` connects to the proxy host/port.
- Check logs for stack traces; loader logs row counts and failures.
- Validate table schemas (types and primary keys) to ensure ON CONFLICT keys align with CSV columns.
- For large loads consider increasing `batch_size` and tuning Postgres settings.

## Example — run full import (local CSV → Cloud DB)

1. Start Cloud SQL Proxy (if required).
2. Ensure `.env` is present with DB connection details.
3. Install deps: `pip install -r requirements.txt`
4. Run import:

```bash
python -c "from sql.load_to_database import load_all_hotel_data_to_database; print(load_all_hotel_data_to_database('intermediate/csv'))"
```

---

If you want, I can:

- Generate a `.env.example` with recommended variable names.
- Add a small example script to call `load_all_hotel_data_to_database`.
- Add simple Airflow run instructions specific to the docker-compose in this repo.
