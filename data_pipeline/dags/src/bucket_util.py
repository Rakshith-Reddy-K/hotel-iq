import os
from google.cloud import storage
from dotenv import load_dotenv
import logging
import time

logger = logging.getLogger(__name__)
load_dotenv()

# Global client and bucket (reuse connections)
_storage_client = None
_bucket = None

def get_bucket():
    """Get bucket with connection reuse"""
    global _storage_client, _bucket
    
    bucket_name = os.getenv('GCS_BUCKET_NAME')
    project_id = os.getenv('GCP_PROJECT_ID')
    
    if not bucket_name:
        raise ValueError("GCS_BUCKET_NAME not found in environment variables")
    if not project_id:
        raise ValueError("GCP_PROJECT_ID not found in environment variables")
    
    # Reuse existing client and bucket
    if _storage_client is None:
        logger.info("Creating new GCS client...")
        _storage_client = storage.Client(project=project_id)
    
    if _bucket is None:
        logger.info(f"Connecting to bucket: {bucket_name}")
        _bucket = _storage_client.bucket(bucket_name)
        if not _bucket.exists():
            raise ValueError(f"Bucket does not exist: {bucket_name}")
    
    return _bucket


def upload_file_to_gcs(file_path, blob_name, chunk_size=5*1024*1024, max_retries=3):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    file_size = os.path.getsize(file_path)
    logger.info(f"Uploading {file_size/1024/1024:.2f} MB to {blob_name}")
    
    # Get bucket once
    bucket = get_bucket()
    blob = bucket.blob(blob_name)
    blob.chunk_size = chunk_size
    
    start_time = time.time()
    
    for attempt in range(max_retries):
        try:
            logger.info(f"Starting upload (attempt {attempt + 1})...")
            
            blob.upload_from_filename(file_path, timeout=120)
            
            elapsed = time.time() - start_time
            speed_mb = (file_size / elapsed / 1024 / 1024) if elapsed > 0 else 0
            
            logger.info(f"Uploaded in {elapsed:.1f}s ({speed_mb:.2f} MB/s)")
            return f"gs://{bucket.name}/{blob.name}"
            
        except Exception as e:
            logger.warning(f"Attempt {attempt + 1} failed: {str(e)}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
            else:
                logger.error(f"Upload failed after {max_retries} attempts")
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
        
        logger.info(f"Downloading {blob_name}...")
        start_time = time.time()
        
        blob.download_to_filename(local_path)
        
        elapsed = time.time() - start_time
        file_size = os.path.getsize(local_path)
        speed_mb = (file_size / elapsed / 1024 / 1024) if elapsed > 0 else 0
        
        logger.info(f"Downloaded in {elapsed:.1f}s ({speed_mb:.2f} MB/s)")
        
        return True
    except Exception as e:
        logger.error(f"Download failed: {e}")
        raise


def list_blobs(prefix=None):
    bucket = get_bucket()
    blobs = list(bucket.list_blobs(prefix=prefix))
    for blob in blobs:
        logger.info(blob.name)
    return blobs