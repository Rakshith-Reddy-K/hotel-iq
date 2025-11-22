# HotelIQ CI/CD Test Suite

Comprehensive test suite for CI/CD integration covering GCP Secret Manager, API endpoints, Langfuse tracking, and Pinecone connectivity.

## Quick Start

### 1. Install Dependencies

```bash
cd backend
pip install -r requirements.txt
```

### 2. Run Tests Locally

```bash
# Run all unit tests (with mocks - no credentials needed)
pytest tests/test_ci.py -v -m "not integration"

# Run all tests including integration tests (requires credentials)
pytest tests/test_ci.py -v

# Run with coverage report
pytest tests/test_ci.py -v --cov=. --cov-report=term-missing
```

## Test Categories

### Unit Tests (Mocked)
- ✅ GCP Secret Manager (mocked)
- ✅ Pinecone connectivity (mocked)
- ✅ Langfuse tracking (mocked)
- ✅ API endpoint validation
- ✅ CORS configuration
- ✅ Environment variable loading

### Integration Tests (Real Credentials)
- 🔐 GCP Secret Manager (requires `GCP_PROJECT_ID` and service account)
- 🔐 Pinecone connectivity (requires `PINECONE_API_KEY`)
- 🔐 Langfuse tracking (requires `LANGFUSE_PUBLIC_KEY` and `LANGFUSE_SECRET_KEY`)

## GitHub Secrets Configuration

To run tests in CI, add these secrets to your GitHub repository:

1. Go to **Settings** → **Secrets and variables** → **Actions**
2. Click **New repository secret**
3. Add each of the following:

| Secret Name | Description | Required |
|-------------|-------------|----------|
| `OPENAI_API_KEY` | OpenAI API key | ✅ Yes |
| `PINECONE_API_KEY` | Pinecone API key | ✅ Yes |
| `HOTEL_INDEX_NAME` | Pinecone hotel index name | ✅ Yes |
| `REVIEWS_INDEX_NAME` | Pinecone reviews index name | ✅ Yes |
| `LANGFUSE_PUBLIC_KEY` | Langfuse public key | ✅ Yes |
| `LANGFUSE_SECRET_KEY` | Langfuse secret key | ✅ Yes |
| `LANGFUSE_HOST` | Langfuse host URL | ✅ Yes |
| `GCP_PROJECT_ID` | Google Cloud project ID | ⚠️ Optional |
| `GCP_SA_KEY` | GCP service account JSON | ⚠️ Optional |

## Running Specific Test Categories

```bash
# Run only GCP tests
pytest tests/test_ci.py -v -k "gcp"

# Run only Pinecone tests
pytest tests/test_ci.py -v -k "pinecone"

# Run only Langfuse tests
pytest tests/test_ci.py -v -k "langfuse"

# Run only API tests
pytest tests/test_ci.py -v -k "api or endpoint"

# Skip integration tests
pytest tests/test_ci.py -v -m "not integration"
```

## CI/CD Pipeline

The GitHub Actions workflow (`.github/workflows/ci.yml`) automatically:
1. Runs on push to `main` or `develop` branches
2. Runs on pull requests
3. Executes unit tests with mocks
4. Optionally runs integration tests if credentials are configured
5. Uploads coverage reports to Codecov

## Test Structure

```
backend/tests/test_ci.py
├── GCP Secret Manager Tests
│   ├── test_gcp_secret_manager_fallback_to_env()
│   ├── test_gcp_secret_manager_connection() [integration]
│   └── test_gcp_secret_manager_mock()
├── Pinecone Connectivity Tests
│   ├── test_pinecone_client_initialization_mock()
│   ├── test_pinecone_client_initialization_real() [integration]
│   ├── test_pinecone_hotel_retrieval_mock()
│   └── test_pinecone_hotel_retrieval_real() [integration]
├── Langfuse Tracking Tests
│   ├── test_langfuse_decorator_functionality()
│   ├── test_langfuse_callback_handler_initialization()
│   └── test_langfuse_connection_real() [integration]
└── API Endpoint Tests
    ├── test_health_endpoint()
    ├── test_chat_endpoint_validation()
    ├── test_chat_endpoint_with_mock()
    ├── test_chat_endpoint_message_length_validation()
    └── test_cors_configuration()
```

## Troubleshooting

### Tests fail with "Module not found"
```bash
# Make sure you're in the backend directory
cd backend
pip install -r requirements.txt
```

### Integration tests are skipped
This is expected if you don't have real credentials configured. Integration tests will automatically skip if credentials are missing.

### GCP tests fail
Make sure `GCP_PROJECT_ID` is set and you have proper authentication configured (either via service account or `gcloud auth`).

## Local Development

For local development without credentials:
```bash
# Run only mocked tests
pytest tests/test_ci.py -v -m "not integration"
```

This will run all tests using mocks, so you don't need any real API keys or credentials.
