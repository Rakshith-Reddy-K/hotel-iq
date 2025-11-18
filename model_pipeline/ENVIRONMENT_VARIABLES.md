# Environment Variables Configuration

This document lists all required environment variables for the HotelIQ application.

## Setup Instructions

Create a `.env` file in the project root with the following variables:

```bash
# Copy and paste this template into your .env file
cp .env.example .env  # If .env.example exists
# OR create .env manually with the variables below
```

---

## Required Environment Variables

### AI/ML Services

```bash
# OpenAI API Key (required for LLM functionality)
OPENAI_API_KEY=sk-your-openai-api-key-here

# Pinecone API Key (required for vector database)
PINECONE_API_KEY=your-pinecone-api-key-here

# Pinecone Index Names (optional, defaults provided)
HOTEL_INDEX_NAME=hoteliq-hotels
REVIEWS_INDEX_NAME=hoteliq-reviews
```

### Google Cloud Platform (GCP) Configuration

```bash
# GCP Project ID
GCP_PROJECT_ID=your-gcp-project-id

# GCS Bucket Name (where processed data is stored)
GCS_BUCKET_NAME=your-gcs-bucket-name

# GCP Region (optional, default: us-central1)
GCP_REGION=us-central1

# GCP Instance Name (optional)
GCP_INSTANCE_NAME=your-instance-name
```

**Important:** GCP service account credentials must be placed at:
```
backend/config/gcp-service-account.json
```

### Data Configuration

```bash
# City name for hotel data (default: boston)
# The application will download processed data for this city from GCS
CITY=boston
```

### Application Configuration

```bash
# Environment mode (development or production)
ENVIRONMENT=development
```

---

## Docker Usage

### Development Mode

```bash
# Start services
docker-compose up

# Or with specific environment variables
CITY=boston docker-compose up
```

### Production Mode

```bash
# Start services in production mode
docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d
```

---

## Data Flow

When the application starts:

1. **Checks** if data files exist locally in `data/processed/{CITY}/`
2. **Downloads** missing files from GCS bucket:
   - `gs://{GCS_BUCKET_NAME}/processed/{city}/hotels.csv`
   - `gs://{GCS_BUCKET_NAME}/processed/{city}/amenities.csv`
   - `gs://{GCS_BUCKET_NAME}/processed/{city}/policies.csv`
   - `gs://{GCS_BUCKET_NAME}/processed/{city}/reviews.csv`
3. **Saves** to local directory: `data/processed/{city}/`
4. **Uses** local files for all agent operations

---

## Verification

To verify your environment variables are set correctly:

```bash
# Check that .env file exists
ls -la .env

# Start the application and check logs
docker-compose up

# Look for these log messages:
# ✅ GCP credentials loaded from: ...
# 🔍 Checking data files for city: boston
# ✅ All data files already exist locally. Skipping download.
# OR
# 🔄 Downloading missing data files: ...
# ✅ Successfully downloaded ...
```

---

## Security Notes

⚠️ **Never commit these files to version control:**
- `.env`
- `backend/config/gcp-service-account.json`

These files should be listed in `.gitignore`.

