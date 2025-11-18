# Google Cloud Platform Deployment Guide for HotelIQ

## Overview
This guide covers deploying HotelIQ chatbot on Google Cloud Platform with secure environment variable management using **Google Secret Manager**.

---

## 📋 Prerequisites

1. **Google Cloud Account**
   - Create account at [cloud.google.com](https://cloud.google.com)
   - Enable billing (free tier available)

2. **Install Google Cloud SDK**
   ```bash
   # macOS
   brew install google-cloud-sdk
   
   # Or download from: https://cloud.google.com/sdk/docs/install
   ```

3. **Authenticate**
   ```bash
   gcloud auth login
   gcloud config set project YOUR_PROJECT_ID
   ```

---

## 🔐 Part 1: Store Secrets in Google Secret Manager

### Step 1: Enable Secret Manager API
```bash
gcloud services enable secretmanager.googleapis.com
```

### Step 2: Create Secrets
```bash
# Create OPENAI_API_KEY secret
echo -n "sk-your-actual-openai-key" | \
  gcloud secrets create OPENAI_API_KEY \
  --data-file=- \
  --replication-policy="automatic"

# Create PINECONE_API_KEY secret
echo -n "pcsk-your-actual-pinecone-key" | \
  gcloud secrets create PINECONE_API_KEY \
  --data-file=- \
  --replication-policy="automatic"

# Create index names
echo -n "hoteliq-hotels" | \
  gcloud secrets create HOTEL_INDEX_NAME \
  --data-file=- \
  --replication-policy="automatic"

echo -n "hoteliq-reviews" | \
  gcloud secrets create REVIEWS_INDEX_NAME \
  --data-file=- \
  --replication-policy="automatic"
```

### Step 3: Verify Secrets
```bash
# List all secrets
gcloud secrets list

# View secret metadata (not the value)
gcloud secrets describe OPENAI_API_KEY

# Access secret value (for testing)
gcloud secrets versions access latest --secret="OPENAI_API_KEY"
```

### Step 4: Grant Access Permissions
```bash
# For Cloud Run service account
PROJECT_NUMBER=$(gcloud projects describe YOUR_PROJECT_ID --format="value(projectNumber)")
SERVICE_ACCOUNT="${PROJECT_NUMBER}-compute@developer.gserviceaccount.com"

# Grant secret accessor role
gcloud secrets add-iam-policy-binding OPENAI_API_KEY \
  --member="serviceAccount:${SERVICE_ACCOUNT}" \
  --role="roles/secretmanager.secretAccessor"

gcloud secrets add-iam-policy-binding PINECONE_API_KEY \
  --member="serviceAccount:${SERVICE_ACCOUNT}" \
  --role="roles/secretmanager.secretAccessor"

gcloud secrets add-iam-policy-binding HOTEL_INDEX_NAME \
  --member="serviceAccount:${SERVICE_ACCOUNT}" \
  --role="roles/secretmanager.secretAccessor"

gcloud secrets add-iam-policy-binding REVIEWS_INDEX_NAME \
  --member="serviceAccount:${SERVICE_ACCOUNT}" \
  --role="roles/secretmanager.secretAccessor"
```

---

## 🚀 Part 2: Update Your Application Code

### Option A: Use Google Secret Manager Client (Recommended)

**Install the library:**
```bash
pip install google-cloud-secret-manager
```

**Update `backend/agents/config.py`:**
```python
"""
Configuration and Global State
===============================

Contains all configuration, paths, and global state variables for the HotelIQ system.
"""

import os
from pathlib import Path
from typing import Any, Dict, List
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings

# Try to load from Secret Manager in production, fall back to .env in development
def get_secret(secret_name: str, default: str = None) -> str:
    """
    Get secret from Google Secret Manager in production, or .env in development.
    
    Args:
        secret_name: Name of the secret
        default: Default value if secret not found
    
    Returns:
        Secret value
    """
    # First try environment variable (for local development)
    value = os.getenv(secret_name)
    if value:
        return value
    
    # In production, try Google Secret Manager
    try:
        from google.cloud import secretmanager
        
        # Get project ID from environment or metadata server
        project_id = os.getenv("GCP_PROJECT_ID") or os.getenv("GOOGLE_CLOUD_PROJECT")
        
        if project_id:
            client = secretmanager.SecretManagerServiceClient()
            name = f"projects/{project_id}/secrets/{secret_name}/versions/latest"
            response = client.access_secret_version(request={"name": name})
            return response.payload.data.decode("UTF-8")
    except Exception as e:
        print(f"⚠️ Could not fetch secret {secret_name} from Secret Manager: {e}")
    
    # Return default if provided
    if default:
        return default
    
    raise ValueError(f"Secret {secret_name} not found in environment or Secret Manager")


# ======================================================
# PATHS
# ======================================================

# BASE_DIR = "Model Development" (one level above backend)
BASE_DIR = Path(__file__).resolve().parent.parent.parent
FRONTEND_DIR = BASE_DIR / "frontend"
BOOKINGS_PATH = BASE_DIR / "booking_requests.json"

# Data paths (Update these paths when adding hotels from different cities)
HOTELS_PATH = BASE_DIR / "processed_boston_hotels.csv"
REVIEWS_PATH = BASE_DIR / "processed_boston_reviews.csv"

# ======================================================
# GLOBAL STATE
# ======================================================

# Track last recommended hotels per thread (for "the first one", "second one")
last_suggestions: Dict[str, List[Dict[str, str]]] = {}

# Track conversation context per thread
conversation_context: Dict[str, Dict[str, Any]] = {}

# Fake "database" for bookings (in memory + JSON file)
bookings_log: List[Dict[str, Any]] = []

# ======================================================
# LLM & EMBEDDINGS
# ======================================================

# Load .env for local development
load_dotenv()

# Get API keys from Secret Manager (production) or .env (development)
OPENAI_API_KEY = get_secret("OPENAI_API_KEY")
PINECONE_API_KEY = get_secret("PINECONE_API_KEY")

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not found")

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.0,
    openai_api_key=OPENAI_API_KEY,
)

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

print("✅ LLM and embeddings configured.")
```

**Update `backend/agents/pinecone_retrieval.py`:**
```python
# At the top of the file, update the get_pinecone_client function:

def get_pinecone_client():
    """Initialize and return Pinecone client."""
    # Import config to use the get_secret function
    from .config import get_secret
    
    api_key = get_secret("PINECONE_API_KEY")
    if not api_key:
        raise ValueError("PINECONE_API_KEY not found")
    return Pinecone(api_key=api_key)


# Update index name retrieval:
index_name = get_secret("HOTEL_INDEX_NAME", default="hoteliq-hotels")
# For reviews:
index_name = get_secret("REVIEWS_INDEX_NAME", default="hoteliq-reviews")
```

### Option B: Use Environment Variables with Secret Manager (Simpler)

Keep your current code as-is and let GCP inject secrets as environment variables at runtime.

---

## 🐳 Part 3: Deployment Options

### **Option 1: Cloud Run (Recommended)** ⭐

Best for serverless, auto-scaling applications.

#### Step 1: Create Dockerfile
```dockerfile
# backend/Dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Expose port
EXPOSE 8080

# Run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
```

#### Step 2: Add requirements
```bash
# Add to backend/requirements.txt
google-cloud-secret-manager>=2.16.0
```

#### Step 3: Deploy to Cloud Run
```bash
cd backend

# Build and deploy in one command
gcloud run deploy hoteliq-backend \
  --source . \
  --region us-central1 \
  --platform managed \
  --allow-unauthenticated \
  --set-secrets="OPENAI_API_KEY=OPENAI_API_KEY:latest,PINECONE_API_KEY=PINECONE_API_KEY:latest,HOTEL_INDEX_NAME=HOTEL_INDEX_NAME:latest,REVIEWS_INDEX_NAME=REVIEWS_INDEX_NAME:latest" \
  --memory 2Gi \
  --cpu 2 \
  --timeout 300 \
  --max-instances 10
```

**Or build Docker image separately:**
```bash
# Build image
gcloud builds submit --tag gcr.io/YOUR_PROJECT_ID/hoteliq-backend

# Deploy
gcloud run deploy hoteliq-backend \
  --image gcr.io/YOUR_PROJECT_ID/hoteliq-backend \
  --region us-central1 \
  --set-secrets="OPENAI_API_KEY=OPENAI_API_KEY:latest,PINECONE_API_KEY=PINECONE_API_KEY:latest"
```

#### Step 4: Access Your API
```bash
# Get the service URL
gcloud run services describe hoteliq-backend --region us-central1 --format="value(status.url)"

# Example: https://hoteliq-backend-xxxxx-uc.a.run.app
```

---

### **Option 2: Google Kubernetes Engine (GKE)**

For more control and complex deployments.

#### Step 1: Create GKE Cluster
```bash
gcloud container clusters create hoteliq-cluster \
  --zone us-central1-a \
  --num-nodes 2 \
  --machine-type e2-medium \
  --enable-autoscaling \
  --min-nodes 1 \
  --max-nodes 5
```

#### Step 2: Create Kubernetes Secret from Google Secret Manager
```bash
# Install External Secrets Operator (recommended)
kubectl apply -f https://raw.githubusercontent.com/external-secrets/external-secrets/main/deploy/crds/bundle.yaml

# Or manually create secrets
kubectl create secret generic hoteliq-secrets \
  --from-literal=OPENAI_API_KEY="$(gcloud secrets versions access latest --secret=OPENAI_API_KEY)" \
  --from-literal=PINECONE_API_KEY="$(gcloud secrets versions access latest --secret=PINECONE_API_KEY)"
```

#### Step 3: Create Deployment
```yaml
# k8s/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: hoteliq-backend
spec:
  replicas: 2
  selector:
    matchLabels:
      app: hoteliq-backend
  template:
    metadata:
      labels:
        app: hoteliq-backend
    spec:
      containers:
      - name: backend
        image: gcr.io/YOUR_PROJECT_ID/hoteliq-backend:latest
        ports:
        - containerPort: 8080
        env:
        - name: GCP_PROJECT_ID
          value: "YOUR_PROJECT_ID"
        envFrom:
        - secretRef:
            name: hoteliq-secrets
        resources:
          requests:
            memory: "1Gi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "1000m"
---
apiVersion: v1
kind: Service
metadata:
  name: hoteliq-backend-service
spec:
  type: LoadBalancer
  selector:
    app: hoteliq-backend
  ports:
  - port: 80
    targetPort: 8080
```

```bash
kubectl apply -f k8s/deployment.yaml
```

---

### **Option 3: App Engine**

Simplest option for traditional web apps.

#### Step 1: Create app.yaml
```yaml
# backend/app.yaml
runtime: python311
entrypoint: gunicorn -w 4 -k uvicorn.workers.UvicornWorker main:app

env_variables:
  GCP_PROJECT_ID: "YOUR_PROJECT_ID"

automatic_scaling:
  min_instances: 1
  max_instances: 10
  target_cpu_utilization: 0.65
```

#### Step 2: Deploy
```bash
cd backend
gcloud app deploy
```

---

## 🔄 Part 4: CI/CD Pipeline

### Option A: Cloud Build

**Create `cloudbuild.yaml`:**
```yaml
steps:
  # Build Docker image
  - name: 'gcr.io/cloud-builders/docker'
    args: ['build', '-t', 'gcr.io/$PROJECT_ID/hoteliq-backend', './backend']
  
  # Push to Container Registry
  - name: 'gcr.io/cloud-builders/docker'
    args: ['push', 'gcr.io/$PROJECT_ID/hoteliq-backend']
  
  # Deploy to Cloud Run
  - name: 'gcr.io/google.com/cloudsdktool/cloud-sdk'
    entrypoint: gcloud
    args:
      - 'run'
      - 'deploy'
      - 'hoteliq-backend'
      - '--image'
      - 'gcr.io/$PROJECT_ID/hoteliq-backend'
      - '--region'
      - 'us-central1'
      - '--platform'
      - 'managed'
      - '--set-secrets'
      - 'OPENAI_API_KEY=OPENAI_API_KEY:latest,PINECONE_API_KEY=PINECONE_API_KEY:latest'

images:
  - 'gcr.io/$PROJECT_ID/hoteliq-backend'
```

**Trigger on GitHub push:**
```bash
gcloud builds triggers create github \
  --repo-name=YOUR_REPO \
  --repo-owner=YOUR_GITHUB_USERNAME \
  --branch-pattern="^main$" \
  --build-config=cloudbuild.yaml
```

### Option B: GitHub Actions

**Create `.github/workflows/deploy.yml`:**
```yaml
name: Deploy to Google Cloud Run

on:
  push:
    branches: [ main ]

env:
  PROJECT_ID: your-gcp-project-id
  SERVICE_NAME: hoteliq-backend
  REGION: us-central1

jobs:
  deploy:
    runs-on: ubuntu-latest
    
    permissions:
      contents: 'read'
      id-token: 'write'
    
    steps:
    - name: Checkout code
      uses: actions/checkout@v3
    
    - name: Authenticate to Google Cloud
      uses: google-github-actions/auth@v1
      with:
        workload_identity_provider: ${{ secrets.WIF_PROVIDER }}
        service_account: ${{ secrets.WIF_SERVICE_ACCOUNT }}
    
    - name: Set up Cloud SDK
      uses: google-github-actions/setup-gcloud@v1
    
    - name: Build and push Docker image
      run: |
        gcloud builds submit --tag gcr.io/$PROJECT_ID/$SERVICE_NAME ./backend
    
    - name: Deploy to Cloud Run
      run: |
        gcloud run deploy $SERVICE_NAME \
          --image gcr.io/$PROJECT_ID/$SERVICE_NAME \
          --region $REGION \
          --platform managed \
          --set-secrets="OPENAI_API_KEY=OPENAI_API_KEY:latest,PINECONE_API_KEY=PINECONE_API_KEY:latest"
```

---

## 🔒 Part 5: Security Best Practices

### 1. **Use Service Accounts**
```bash
# Create dedicated service account
gcloud iam service-accounts create hoteliq-sa \
  --display-name="HotelIQ Service Account"

# Grant only necessary permissions
gcloud projects add-iam-policy-binding YOUR_PROJECT_ID \
  --member="serviceAccount:hoteliq-sa@YOUR_PROJECT_ID.iam.gserviceaccount.com" \
  --role="roles/secretmanager.secretAccessor"
```

### 2. **Enable VPC Service Controls** (optional)
```bash
gcloud access-context-manager perimeters create hoteliq-perimeter \
  --title="HotelIQ Perimeter" \
  --resources=projects/YOUR_PROJECT_NUMBER \
  --restricted-services=secretmanager.googleapis.com
```

### 3. **Set up Secret Rotation**
```bash
# Create rotation schedule (manual process)
gcloud secrets update OPENAI_API_KEY \
  --next-rotation-time="2024-12-31T00:00:00Z" \
  --rotation-period="90d"
```

### 4. **Enable Audit Logging**
```bash
# View who accessed secrets
gcloud logging read "resource.type=secretmanager.googleapis.com/Secret" \
  --limit 50 \
  --format json
```

---

## 📊 Part 6: Monitoring & Logging

### Enable Cloud Monitoring
```bash
# View logs
gcloud logging read "resource.type=cloud_run_revision AND resource.labels.service_name=hoteliq-backend" \
  --limit 50

# Create alert for errors
gcloud alpha monitoring policies create \
  --notification-channels=CHANNEL_ID \
  --display-name="HotelIQ Error Alert" \
  --condition-display-name="Error Rate" \
  --condition-threshold-value=10
```

---

## 💰 Part 7: Cost Optimization

### Cloud Run Tips
- Use minimum instances = 0 (serverless)
- Set max instances to prevent runaway costs
- Use `--cpu-throttling` for non-latency-sensitive workloads

### Estimate Costs
```bash
# Get current usage
gcloud billing accounts list
gcloud billing projects describe YOUR_PROJECT_ID
```

**Typical monthly costs for moderate usage:**
- Cloud Run: $5-20
- Secret Manager: $1-2
- Container Registry: $2-5
- **Total: ~$10-30/month**

---

## ✅ Quick Start Checklist

- [ ] Enable required APIs
- [ ] Create secrets in Secret Manager
- [ ] Update `config.py` to use `get_secret()` function
- [ ] Create Dockerfile
- [ ] Test locally with Docker
- [ ] Deploy to Cloud Run
- [ ] Set up custom domain (optional)
- [ ] Configure CI/CD pipeline
- [ ] Set up monitoring alerts
- [ ] Enable backup/disaster recovery

---

## 🆘 Troubleshooting

### Secret Access Denied
```bash
# Check IAM permissions
gcloud secrets get-iam-policy OPENAI_API_KEY

# Grant access
gcloud secrets add-iam-policy-binding OPENAI_API_KEY \
  --member="serviceAccount:YOUR_SA@PROJECT.iam.gserviceaccount.com" \
  --role="roles/secretmanager.secretAccessor"
```

### Application Can't Find Secrets
```python
# Test secret access
from google.cloud import secretmanager
client = secretmanager.SecretManagerServiceClient()
name = "projects/YOUR_PROJECT_ID/secrets/OPENAI_API_KEY/versions/latest"
response = client.access_secret_version(request={"name": name})
print(response.payload.data.decode("UTF-8"))
```

### Container Build Fails
```bash
# Build locally first
docker build -t hoteliq-backend ./backend
docker run -p 8080:8080 hoteliq-backend

# Check logs
gcloud builds log BUILD_ID
```

---

## 📚 Additional Resources

- [Google Cloud Run Documentation](https://cloud.google.com/run/docs)
- [Secret Manager Best Practices](https://cloud.google.com/secret-manager/docs/best-practices)
- [Cloud Build Configuration](https://cloud.google.com/build/docs/configuring-builds/create-basic-configuration)
- [Python on Google Cloud](https://cloud.google.com/python/docs)

---

## 🎯 Next Steps

1. Start with Cloud Run deployment (simplest)
2. Set up CI/CD with Cloud Build
3. Configure custom domain with Cloud DNS
4. Add Cloud CDN for static assets
5. Implement Cloud Armor for DDoS protection
6. Set up Cloud SQL if you need persistent database

Need help with any specific step? Let me know!

