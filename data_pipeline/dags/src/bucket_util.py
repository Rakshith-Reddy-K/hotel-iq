import os
from google.cloud import storage
from dotenv import load_dotenv

load_dotenv()

def get_bucket():
    bucket_name = os.getenv('GCS_BUCKET_NAME')
    project_id = os.getenv('PROJECT_ID')
    if not bucket_name:
        raise ValueError("GCS_BUCKET_NAME not found in environment variables")
    if not project_id:
        raise ValueError("PROJECT_ID not found in environment variables")

    client = storage.Client(project=project_id)
    bucket = client.bucket(bucket_name)
    
    if not bucket.exists():
        raise ValueError("Bucket does not exist")
    return bucket

def upload_file_to_gcs(local_path, blob_name):
    bucket = get_bucket()
    blob = bucket.blob(blob_name)
    blob.upload_from_filename(local_path)

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