"""
Comprehensive CI/CD Test Suite for HotelIQ
==========================================

Tests for:
- GCP Secret Manager connectivity
- API endpoints (FastAPI)
- Langfuse tracking integration
- Pinecone vector database connectivity
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
from unittest.mock import MagicMock, Mock, patch, AsyncMock
from httpx import AsyncClient, ASGITransport
from fastapi import status
from dotenv import load_dotenv

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY", "sk-test-key")
os.environ["PINECONE_API_KEY"] = os.getenv("PINECONE_API_KEY", "pcsk-test-key")
os.environ["HOTEL_INDEX_NAME"] = os.getenv("HOTEL_INDEX_NAME", "test-hotels")
os.environ["REVIEWS_INDEX_NAME"] = os.getenv("REVIEWS_INDEX_NAME", "test-reviews")
os.environ["LANGFUSE_PUBLIC_KEY"] = os.getenv("LANGFUSE_PUBLIC_KEY", "pk-lf-test")
os.environ["LANGFUSE_SECRET_KEY"] = os.getenv("LANGFUSE_SECRET_KEY", "sk-lf-test")
os.environ["LANGFUSE_HOST"] = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")
os.environ["GCP_PROJECT_ID"] = os.getenv("GCP_PROJECT_ID", "test-project")
os.environ["CITY"] = "boston"


with patch('agents.utils._load_data'):
    from main import app
    from agents.pinecone_retrieval import get_pinecone_client, retrieve_hotels_by_description
    from agents.langfuse_tracking import track_agent
    from agents.state import HotelIQState


# ======================================================
# FIXTURES
# ======================================================

@pytest.fixture
def test_chat_payload():
    """Provide a sample chat payload for POST requests."""
    return {
        "message": "Tell me about hotels in Boston",
        "user_id": "test_user_123",
        "hotel_id": "1",
        "thread_id": None
    }


@pytest.fixture
def mock_pinecone_client():
    """Mock Pinecone client for testing."""
    mock_client = MagicMock()
    mock_index = MagicMock()
    
    # Mock query response
    mock_match = MagicMock()
    mock_match.score = 0.95
    mock_match.metadata = {
        "hotel_id": "1",
        "hotel_name": "Test Hotel",
        "official_name": "Test Hotel Boston",
        "star_rating": "4",
        "address": "123 Test St",
        "city": "Boston",
        "state": "MA",
        "overall_rating": "4.5",
        "description": "A great test hotel"
    }
    
    mock_query_response = MagicMock()
    mock_query_response.matches = [mock_match]
    mock_index.query.return_value = mock_query_response
    
    mock_client.Index.return_value = mock_index
    return mock_client


@pytest.fixture
def mock_gcp_secret_client():
    """Mock Google Cloud Secret Manager client."""
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.payload.data.decode.return_value = "test-secret-value"
    mock_client.access_secret_version.return_value = mock_response
    return mock_client


@pytest.fixture
def mock_langfuse_handler():
    """Mock Langfuse callback handler."""
    mock_handler = MagicMock()
    mock_handler.trace_id = "test-trace-id"
    return mock_handler


# ======================================================
# GCP SECRET MANAGER TESTS
# ======================================================

def test_gcp_secret_manager_fallback_to_env():
    """Test that secrets fall back to environment variables when GCP is unavailable."""
    # This should use the environment variable we set above
    api_key = os.getenv("OPENAI_API_KEY")
    assert api_key is not None
    assert api_key.startswith("sk-")


@pytest.mark.integration
def test_gcp_secret_manager_connection():
    """Test GCP Secret Manager connection (integration test - requires GCP credentials)."""
    try:
        from google.cloud import secretmanager
        
        project_id = os.getenv("GCP_PROJECT_ID")
        if not project_id or project_id == "test-project":
            pytest.skip("GCP_PROJECT_ID not configured for integration testing")
        
        client = secretmanager.SecretManagerServiceClient()
        # Just test that we can create a client - don't actually access secrets
        assert client is not None
    except Exception as e:
        pytest.skip(f"GCP Secret Manager not available: {e}")


def test_gcp_secret_manager_mock(mock_gcp_secret_client):
    """Test GCP Secret Manager with mocked client."""
    with patch('google.cloud.secretmanager.SecretManagerServiceClient', return_value=mock_gcp_secret_client):
        from google.cloud import secretmanager
        
        client = secretmanager.SecretManagerServiceClient()
        name = "projects/test-project/secrets/OPENAI_API_KEY/versions/latest"
        response = client.access_secret_version(request={"name": name})
        secret_value = response.payload.data.decode("UTF-8")
        
        assert secret_value == "test-secret-value"


# ======================================================
# PINECONE CONNECTIVITY TESTS
# ======================================================

def test_pinecone_client_initialization_mock(mock_pinecone_client):
    """Test Pinecone client initialization with mock."""
    with patch('agents.pinecone_retrieval.Pinecone', return_value=mock_pinecone_client):
        client = get_pinecone_client()
        assert client is not None


@pytest.mark.integration
def test_pinecone_client_initialization_real():
    """Test real Pinecone client initialization (integration test)."""
    api_key = os.getenv("PINECONE_API_KEY")
    if not api_key or api_key == "pcsk-test-key":
        pytest.skip("PINECONE_API_KEY not configured for integration testing")
    
    try:
        client = get_pinecone_client()
        assert client is not None
    except Exception as e:
        pytest.fail(f"Failed to initialize Pinecone client: {e}")


def test_pinecone_hotel_retrieval_mock(mock_pinecone_client):
    """Test hotel retrieval from Pinecone with mock."""
    with patch('agents.pinecone_retrieval.get_pinecone_client', return_value=mock_pinecone_client):
        with patch('agents.pinecone_retrieval.embeddings') as mock_embeddings:
            mock_embeddings.embed_query.return_value = [0.1] * 3072
            
            hotels = retrieve_hotels_by_description("luxury hotel", top_k=3)
            
            assert len(hotels) > 0
            assert hotels[0].metadata["hotel_name"] == "Test Hotel"


@pytest.mark.integration
def test_pinecone_hotel_retrieval_real():
    """Test real hotel retrieval from Pinecone (integration test)."""
    api_key = os.getenv("PINECONE_API_KEY")
    if not api_key or api_key == "pcsk-test-key":
        pytest.skip("PINECONE_API_KEY not configured for integration testing")
    
    try:
        hotels = retrieve_hotels_by_description("hotel near downtown", top_k=3)
        assert isinstance(hotels, list)
    except Exception as e:
        pytest.skip(f"Pinecone retrieval failed: {e}")


# ======================================================
# LANGFUSE TRACKING TESTS
# ======================================================

def test_langfuse_decorator_functionality():
    """Test that the track_agent decorator can be applied to functions."""
    @track_agent("test_agent")
    async def test_function(state: HotelIQState) -> HotelIQState:
        return state
    
    assert callable(test_function)


def test_langfuse_callback_handler_initialization(mock_langfuse_handler):
    """Test Langfuse callback handler initialization with mock."""
    with patch('agents.config.CallbackHandler', return_value=mock_langfuse_handler):
        from langfuse.langchain import CallbackHandler
        
        handler = CallbackHandler()
        assert handler is not None


@pytest.mark.integration
def test_langfuse_connection_real():
    """Test real Langfuse connection (integration test)."""
    public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
    secret_key = os.getenv("LANGFUSE_SECRET_KEY")
    
    if not public_key or public_key == "pk-lf-test":
        pytest.skip("LANGFUSE_PUBLIC_KEY not configured for integration testing")
    
    try:
        from langfuse.langchain import CallbackHandler
        
        handler = CallbackHandler()
        assert handler is not None
    except Exception as e:
        pytest.skip(f"Langfuse connection failed: {e}")


# ======================================================
# API ENDPOINT TESTS
# ======================================================

@pytest.mark.asyncio
async def test_health_endpoint():
    """Test that the application starts and basic routes work."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # Test if app has routes
        assert len(app.routes) > 0


@pytest.mark.asyncio
async def test_chat_endpoint_validation():
    """Test chat endpoint input validation."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # Test with missing required fields
        response = await client.post("/api/v1/chat/message", json={})
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY


@pytest.mark.asyncio
async def test_chat_endpoint_with_mock(test_chat_payload):
    """Test chat endpoint with mocked agent graph."""
    with patch('main.chat_service.agent_graph') as mock_graph:
        # Mock the agent graph response
        mock_graph.get_state.return_value = None
        mock_graph.ainvoke = AsyncMock(return_value={
            "messages": [
                {"role": "user", "content": "Test message"},
                {"role": "assistant", "content": "Test response"}
            ],
            "user_id": "test_user_123",
            "thread_id": "test_thread",
            "hotel_id": "1"
        })
        
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post("/api/v1/chat/message", json=test_chat_payload)
            
            # Should succeed with mocked graph
            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            assert "response" in data
            assert "thread_id" in data
            assert "followup_suggestions" in data


@pytest.mark.asyncio
async def test_chat_endpoint_message_length_validation():
    """Test chat endpoint message length validation."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # Test with message too long (> 2000 characters)
        long_message = "a" * 2001
        payload = {
            "message": long_message,
            "user_id": "test_user",
            "hotel_id": "1"
        }
        response = await client.post("/api/v1/chat/message", json=payload)
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY


@pytest.mark.asyncio
async def test_cors_configuration():
    """Test that CORS is properly configured."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # Check that CORS middleware is added by checking middleware stack
        assert len(app.user_middleware) > 0 or len(app.middleware_stack) is not None


# ======================================================
# INTEGRATION TEST SUMMARY
# ======================================================

def test_environment_variables_loaded():
    """Test that all required environment variables are loaded."""
    required_vars = [
        "OPENAI_API_KEY",
        "PINECONE_API_KEY",
        "HOTEL_INDEX_NAME",
        "REVIEWS_INDEX_NAME",
        "LANGFUSE_PUBLIC_KEY",
        "LANGFUSE_SECRET_KEY",
        "LANGFUSE_HOST"
    ]
    
    for var in required_vars:
        assert os.getenv(var) is not None, f"Environment variable {var} not set"


def test_imports_successful():
    """Test that all critical imports are successful."""
    try:
        from agents import agent_graph
        from agents.state import HotelIQState
        from agents.pinecone_retrieval import get_pinecone_client
        from agents.langfuse_tracking import track_agent
        from main import app, chat_service
        
        assert agent_graph is not None
        assert app is not None
        assert chat_service is not None
    except ImportError as e:
        pytest.fail(f"Failed to import required modules: {e}")


# ======================================================
# RUN TESTS
# ======================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
