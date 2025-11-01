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
    logger.info(f"Connected to bucket: {bucket.name}")
    if not bucket.exists():
        raise ValueError("Bucket does not exist")
    return bucket

def upload_file_to_gcs(file_path, blob_name):
    logger.info(f"Starting GCS upload: {file_path} -> {blob_name}")
    if not os.path.exists(file_path):
        logger.error(f"File not found: {file_path}")
        raise FileNotFoundError(f"File not found: {file_path}")
    
    file_size = os.path.getsize(file_path)
    logger.info(f"Local file size: {file_size:,} bytes")
    
    try:
        bucket = get_bucket()
        blob = bucket.blob(blob_name)
        logger.info(f"Uploading to: gs://{bucket.name}/{blob_name}")
        blob.upload_from_filename(file_path)
        logger.info(f"Upload successful - Size: {blob.size:,} bytes")
        logger.info(f"GCS path: gs://{bucket.name}/{blob.name}")
        
        return f"gs://{bucket.name}/{blob.name}"
        
    except Exception as e:
        logger.error(f"Upload failed: {e}")

def download_file_from_gcs(blob_name, local_path):
    try:
        if not blob_name or not local_path:
            raise ValueError("blob_name and local_path are required")
        bucket = get_bucket()
        blob = bucket.blob(blob_name)
        if not blob.exists():
            raise FileNotFoundError(f"Blob '{blob_name}' not found")
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        logger.info(f"Downloading from: gs://{bucket.name}/{blob_name}")
        blob.download_to_filename(local_path)
        logger.info(f"Download successful: {blob_name} -> {local_path}")
        logger.info(f"Local path: {local_path}")

        return True 
    except Exception as e:
        logger.info(f"Download failed: {e}")
        return False

def list_blobs(prefix=None):
    bucket = get_bucket()
    blobs = list(bucket.list_blobs(prefix=prefix))
    for blob in blobs:
        logger.info(blob.name)
    return blobs