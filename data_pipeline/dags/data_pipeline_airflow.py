from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from datetime import datetime, timedelta

from src.extract import extract_metadata,extract_reviews,join_metadata_and_reviews
from src.transform import preprocess_reviews_data
from src.validation import validate_data
from data_pipeline.sql.load_to_database import load_all_hotel_data_to_database
from sql.queries import create_all_tables

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
    
data_validation = PythonOperator(
        task_id='validate_data',
        python_callable=validate_data,
)

preprocess = PythonOperator(
    task_id='preprocess_reviews',
    python_callable=preprocess_reviews_data,
)

create_tables_in_database = PythonOperator(
    task_id='create_tables',
    python_callable=create_all_tables,
)

save_data_to_database = PythonOperator(
    task_id='save_metadata_to_database',
    python_callable=load_all_hotel_data_to_database,
)

[extract_metadata_task, extract_reviews_task] >> preprocess >> data_validation >> create_tables_in_database >> save_data_to_database
