from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from datetime import datetime, timedelta
import os
import sys


# Import our custom functions
from src.extract import extract_metadata, extract_reviews
from src.transform import (
    extract_reviews_based_on_city, 
    compute_aggregate_ratings, 
    enrich_hotels_perplexity,
    merge_sql_tables
)
from sql.load_to_database import load_all_hotel_data_to_database
from sql.queries import create_all_tables

# Default arguments
default_args = {
    'owner': 'hotel-iq-team',
    'depends_on_past': False,
    'start_date': datetime(2025, 1, 1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 0,
    'retry_delay': timedelta(minutes=5),
    'catchup': False
}

# DAG definition
dag = DAG(
    'hotel_data_pipeline',
    default_args=default_args,
    description='Hotel Data pipeline',
    max_active_runs=1,
    tags=['hotel', 'data-pipeline', 'perplexity', 'reviews']
)

# Task 1: Extract hotel metadata from offering.txt
extract_hotels_task = PythonOperator(
    task_id='extract_hotels',
    python_callable=extract_metadata,
    op_kwargs={
        'city': 'Boston',
        'sample_size': 50,
        'random_seed': 42,
        'offering_path': 'data/raw/hotels.txt',
        'output_dir': 'output'
    },
    dag=dag,
    retries=1
)

# Task 2: Extract reviews from review.txt
extract_reviews_task = PythonOperator(
    task_id='extract_reviews',
    python_callable=extract_reviews,
    op_kwargs={
        'reviews_path': 'data/raw/reviews.txt',
        'output_dir': 'output'
    },
    dag=dag,
    retries=1
)

# Task 3: Filter reviews for city hotels
filter_city_reviews_task = PythonOperator(
    task_id='extract_reviews_based_on_city',
    python_callable=extract_reviews_based_on_city,
    op_kwargs={
        'city': 'Boston',
        'output_dir': 'output'
    },
    dag=dag,
    retries=1
)

# Task 4: Compute aggregate ratings (parallel with enrichment)
compute_ratings_task = PythonOperator(
    task_id='compute_aggregate_ratings',
    python_callable=compute_aggregate_ratings,
    op_kwargs={
        'city': 'Boston',
        'output_dir': 'output'
    },
    dag=dag,
    retries=1
)

# Task 5: Enrich hotels with Perplexity API (parallel with ratings)
enrich_hotels_task = PythonOperator(
    task_id='get_metadata_hotel_perplexity',
    python_callable=enrich_hotels_perplexity,
    op_kwargs={
        'city': 'Boston',
        'output_dir': 'output',
        'delay_seconds': 12,
        'max_hotels': None  # Process all hotels
    },
    dag=dag,
    retries=3,  # More retries for API calls
    retry_delay=timedelta(minutes=2)
)

# Task 6: Merge all data into SQL-ready tables
merge_tables_task = PythonOperator(
    task_id='merge_sql_tables',
    python_callable=merge_sql_tables,
    op_kwargs={
        'city': 'Boston',
        'output_dir': 'output'
    },
    dag=dag,
    retries=1
)



# Task 7: Create database tables
create_tables_task = PythonOperator(
    task_id='create_tables',
    python_callable=create_all_tables,
    dag=dag,
    retries=1
)

# Task 8: Load data to database
load_to_db_task = PythonOperator(
    task_id='load_to_db',
    python_callable=load_all_hotel_data_to_database,
    op_kwargs={
        'csv_directory': 'output'
    },
    dag=dag,
    retries=2
)

# Define task dependencies
# Step 1: Extract raw data
[extract_hotels_task, extract_reviews_task] >> filter_city_reviews_task

# Step 2: Parallel processing
filter_city_reviews_task >> [compute_ratings_task, enrich_hotels_task]

# Step 3: Merge and Create tables
[compute_ratings_task, enrich_hotels_task] >> merge_tables_task >> create_tables_task

# Step 4: Database operations
create_tables_task >> load_to_db_task
