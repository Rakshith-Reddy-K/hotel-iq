from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from datetime import datetime, timedelta

from src.extract import extract_metadata,extract_reviews,join_metadata_and_reviews
from src.transform import preprocess_reviews_data
from src.validation import validate_data
from src.load_to_database import save_reviews_to_db,save_metadata_to_db,save_raw_data_to_gcs

default_args = {
    'owner': 'iq-team',
    'retries': 1,
    'start_date': datetime(2025, 10, 27),
    'retry_delay': timedelta(minutes=5)
}

dag = DAG(
    'hotel_data_pipeline',
    default_args=default_args,
    description='Data pipeline for hotel and reviews data',
    schedule_interval='@daily',
    catchup=False
)

extract_metadata_task = PythonOperator(
    task_id='extract_metadata',
    python_callable=extract_metadata,
    dag=dag
)

extract_reviews_task = PythonOperator(
    task_id='extract_reviews_data',
    python_callable=extract_reviews,
    dag=dag
)

join_data = PythonOperator(
        task_id='join_metadata_and_reviews',
        python_callable=join_metadata_and_reviews,
)
    
data_validation = PythonOperator(
        task_id='validate_data',
        python_callable=validate_data,
)

preprocess = PythonOperator(
    task_id='preprocess_reviews',
    python_callable=preprocess_reviews_data,
)

save_raw_data = PythonOperator(
    task_id='save_raw_data_to_gcs',
    python_callable=save_raw_data_to_gcs,
)

save_metadata_to_database = PythonOperator(
    task_id='save_metadata_to_database',
    python_callable=save_metadata_to_db,
)

save_reviews_to_database = PythonOperator(
    task_id='save_reviews_to_database',
    python_callable=save_reviews_to_db,
)

# Step 1: Extract metadata and reviews in parallel
[extract_metadata_task, extract_reviews_task] >> join_data

# Step 2: After joining, save raw data and preprocess in parallel
join_data >> [save_raw_data, preprocess]

# Step 3: Validate after preprocessing
preprocess >> data_validation

# Step 4: After validation passes, save to database in parallel
data_validation >> [save_metadata_to_database, save_reviews_to_database]