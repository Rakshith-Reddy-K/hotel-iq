#!/bin/bash
set -e

REPO_PATH="${REPO_PATH:-/opt/airflow/hotel-iq}"
GCS_BUCKET_NAME="${GCS_BUCKET_NAME}"
GCS_RAW_PATH="${GCS_RAW_PATH:-raw}"
GCS_PROCESSED_PATH="${GCS_PROCESSED_PATH:-processed}"

GIT_USER_NAME="${GIT_USER_NAME:-Airflow Bot}"
GIT_USER_EMAIL="${GIT_USER_EMAIL:-airflow@example.com}"
GIT_TOKEN="${GIT_TOKEN}" 
echo "Starting data versioning..."

echo "Configuring git..."
git config --global user.name "$GIT_USER_NAME"
git config --global user.email "$GIT_USER_EMAIL"

if [ -n "$GIT_TOKEN" ]; then
    git config --global credential.helper store
    echo "https://${GIT_TOKEN}@github.com" > ~/.git-credentials
fi

cd "$REPO_PATH"

echo "Checking directories..."
[ ! -d "$REPO_PATH/data/raw" ] && mkdir -p "$REPO_PATH/data_pipeline/data/raw" && echo "Created data_pipeline/data/raw/"
[ ! -d "$REPO_PATH/data/processed" ] && mkdir -p "$REPO_PATH/data_pipeline/data/processed" && echo "Created data_pipeline/data/processed/"

# echo "Downloading from GCS..."
# gsutil -m -o "GSUtil:parallel_process_count=1" rsync -r \
#     "gs://$GCS_BUCKET_NAME/$GCS_RAW_PATH" "$REPO_PATH/data_pipeline/data/raw"

# gsutil -m -o "GSUtil:parallel_process_count=1" rsync -r \
#     "gs://$GCS_BUCKET_NAME/$GCS_PROCESSED_PATH" "$REPO_PATH/data_pipeline/data/processed"

echo "Tracking with DVC..."
dvc add data_pipeline/data/raw/
dvc add data_pipeline/data/processed/
dvc push

echo "Committing to Git..."
echo "Pulling latest changes..."
git checkout main
git fetch origin
git rebase origin/main
TIMESTAMP=$(date +"%Y-%m-%d_%H:%M:%S")
git add data_pipeline/data/raw.dvc data_pipeline/data/processed.dvc

if git diff --staged --quiet; then
    echo "No changes to commit"
else
    git commit -m "chore: update data versions - $TIMESTAMP"
    git push origin main || echo "Push failed"
fi

echo "Data versioning complete!"