from airflow import DAG
from airflow.operators.python_operator import PythonOperator,BranchPythonOperator
from airflow.operators.empty import EmptyOperator
from airflow.operators.bash import BashOperator

from datetime import datetime, timedelta


# Import our custom functions
from src.transform import ( 
    compute_aggregate_ratings, 
    prepare_hotel_data_for_db
)
from sql.queries import create_all_tables,bulk_insert_from_csvs
from src.filtering import check_if_filtering_needed,filter_all_city_hotels,filter_all_city_reviews
from src.batch_selection import select_next_batch,filter_reviews_for_batch
from src.accumulated import append_batch_to_accumulated
from src.state_management import update_processing_state
from src.prompt import enrich_hotels_gemini

#Configs
CITY = 'Boston'
BATCH_SIZE = 25

# Default arguments
default_args = {
    'owner': 'hotel-iq-team',
    'depends_on_past': False,
    'start_date': datetime(2025, 1, 1),
    'email': ['rakshithreddy444@gmail.com'],  
    'email_on_failure': True,
    'email_on_retry': True,
    'email_on_success': True,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

# DAG definition
dag = DAG(
    'hotel_data_pipeline',
    default_args=default_args,
    description='Hotel Data pipeline',
    max_active_runs=1,
    schedule_interval=None,
    catchup=False
)

def check_filtering_branch(**context):
    result = check_if_filtering_needed(city=CITY)
    
    if result == 'skip_filtering':
        return 'skip_filtering'
    else:
        return 'filter_all_city_hotels'

# Check if filtering is needed or skip to batch selection
check_filtering_task = BranchPythonOperator(
    task_id='check_if_filtering_needed',
    python_callable=check_filtering_branch,
    dag=dag,
)

# Empty operator for skipping filtering step
skip_filtering_task = EmptyOperator(
    task_id='skip_filtering',
    dag=dag,
)

# Filter raw hotel data by city from master dataset
filter_hotels_task = PythonOperator(
    task_id='filter_all_city_hotels',
    python_callable=filter_all_city_hotels,
    op_kwargs={
        'city': CITY,
        'all_hotels_path': 'data/raw/hotels.txt'
    },
    dag=dag,
)

# Filter raw reviews data by city from master dataset
filter_reviews_task = PythonOperator(
    task_id='filter_all_city_reviews',
    python_callable=filter_all_city_reviews,
    op_kwargs={
        'city': CITY,
        'all_reviews_path': 'data/raw/reviews.txt'
    },
    dag=dag,
)

# Join point after filtering check branches
join_filtering_task = EmptyOperator(
    task_id='join_after_filtering_check',
    trigger_rule='none_failed_min_one_success',
    dag=dag,
)

# Select next batch of hotels to process based on state
select_batch_task = PythonOperator(
    task_id='select_next_batch',
    python_callable=select_next_batch,
    op_kwargs={
        'city': CITY,
        'batch_size': BATCH_SIZE
    },
    dag=dag,
)

# Filter reviews for the current batch of hotels
filter_batch_reviews_task = PythonOperator(
    task_id='filter_reviews_for_batch',
    python_callable=filter_reviews_for_batch,
    op_kwargs={'city': CITY},
    dag=dag,
)

# Compute aggregate ratings from reviews for batch hotels
compute_ratings_task = PythonOperator(
    task_id='compute_aggregate_ratings',
    python_callable=compute_aggregate_ratings,
    op_kwargs={
        'city': CITY
    },
    dag=dag,
)

# Enrich hotel data with metadata from Gemini API
enrich_hotels_task = PythonOperator(
    task_id='get_hotel_enrichment_data',
    python_callable=enrich_hotels_gemini,
    op_kwargs={
        'city': CITY,
        'delay_seconds': 5,
        'max_hotels': None
    },
    dag=dag,
)

# Merge and prepare final database-ready CSV files
prepare_hotel_data_for_db_task = PythonOperator(
    task_id='prepare_data_for_db',
    python_callable=prepare_hotel_data_for_db,
    op_kwargs={
        'city': CITY,
    },
    dag=dag,
)

# Create all required database tables if they don't exist
create_tables_task = PythonOperator(
    task_id='create_tables',
    python_callable=create_all_tables,
    dag=dag,
)

# Bulk insert batch data into PostgreSQL database
load_to_db_task = PythonOperator(
    task_id='insert_data_to_db',
    python_callable=bulk_insert_from_csvs,
    op_kwargs={
        'csv_directory': f'data/processed/{CITY}'
    },
    dag=dag,
)

# Append current batch to accumulated results in GCS
accumulate_batch_task = PythonOperator(
    task_id='accumulate_batch',
    python_callable=append_batch_to_accumulated,
    op_kwargs={
        'city': CITY
    },
    dag=dag,
)

# Update processing state with completed batch information
update_state_task = PythonOperator(
    task_id='update_state',
    python_callable=update_processing_state,
    op_kwargs={
        'city': CITY
    },
    dag=dag,
)

push_version_data_to_git_task = BashOperator(
    task_id='push_version_data_to_git',
    bash_command='/opt/airflow/hotel-iq/data_pipeline/scripts/version_data.sh ',
    dag=dag
)

# Task Dependencies
check_filtering_task >> [filter_hotels_task, skip_filtering_task]

# Filtering path
filter_hotels_task >> filter_reviews_task >> join_filtering_task

# Skip path
skip_filtering_task >> join_filtering_task

join_filtering_task >> select_batch_task >> filter_batch_reviews_task 

filter_batch_reviews_task >> [compute_ratings_task, enrich_hotels_task] >> prepare_hotel_data_for_db_task

prepare_hotel_data_for_db_task >> create_tables_task >> load_to_db_task

load_to_db_task >> accumulate_batch_task >> update_state_task >> push_version_data_to_git_task