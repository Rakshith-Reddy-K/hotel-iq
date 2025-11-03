import os
from google.cloud import storage
from dotenv import load_dotenv
import logging
from google.api_core import retry, exceptions
import time

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

def upload_file_to_gcs(file_path, blob_name, chunk_size=5*1024*1024, max_retries=3):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    file_size = os.path.getsize(file_path)
    logger.info(f"Uploading {file_size:,} bytes to {blob_name}")
    
    for attempt in range(max_retries):
        try:
            bucket = get_bucket()
            blob = bucket.blob(blob_name)
            
            if file_size > 10 * 1024 * 1024:
                blob.chunk_size = chunk_size
            
            blob.upload_from_filename(
                file_path,
                timeout=600,
                retry=retry.Retry(deadline=600)
            )
            
            logger.info(f"Upload successful: gs://{bucket.name}/{blob.name}")
            return f"gs://{bucket.name}/{blob.name}"
            
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
            else:
                logger.error(f"Upload failed: {e}")
                raise

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
        raise e

def list_blobs(prefix=None):
    bucket = get_bucket()
    blobs = list(bucket.list_blobs(prefix=prefix))
    for blob in blobs:
        logger.info(blob.name)
    return blobs