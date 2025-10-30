import os
from google.cloud import storage
from dotenv import load_dotenv
import logging

logger = logging.getLogger(__name__)
load_dotenv()

def get_bucket():
    bucket_name = os.getenv('GCS_BUCKET_NAME')
    GCP_PROJECT_ID = os.getenv('GCP_PROJECT_ID')
    if not bucket_name:
        raise ValueError("GCS_BUCKET_NAME not found in environment variables")
    if not GCP_PROJECT_ID:
        raise ValueError("GCP_PROJECT_ID not found in environment variables")

    client = storage.Client(project=GCP_PROJECT_ID)
    bucket = client.bucket(bucket_name)
    
    if not bucket.exists():
        raise ValueError("Bucket does not exist")
    return bucket

def upload_file_to_gcs(local_path, blob_name):
    logger.info(f"Starting GCS upload: {local_path} -> {blob_name}")
    
    if not os.path.exists(local_path):
        logger.error(f"File not found: {local_path}")
        raise FileNotFoundError(f"File not found: {local_path}")
    
    file_size = os.path.getsize(local_path)
    logger.info(f"Local file size: {file_size:,} bytes")
    
    try:
        bucket = get_bucket()
        logger.info(f"Connected to bucket: {bucket.name}")
        
        blob = bucket.blob(blob_name)
        logger.info(f"Uploading to: gs://{bucket.name}/{blob_name}")
        
        blob.upload_from_filename(local_path)
        
        blob.reload()
        logger.info(f"Upload successful - Size: {blob.size:,} bytes")
        logger.info(f"GCS path: gs://{bucket.name}/{blob.name}")
        
        return f"gs://{bucket.name}/{blob.name}"
        
    except Exception as e:
        logger.error(f"Upload failed: {e}")
        logger.exception("Full error:")

def download_file_from_gcs(blob_name, local_path):
    bucket = get_bucket()
    blob = bucket.blob(blob_name)
    blob.download_to_filename(local_path)

def list_blobs(prefix=None):
    bucket = get_bucket()
    blobs = list(bucket.list_blobs(prefix=prefix))
    for blob in blobs:
        print(blob.name)
    return blobs