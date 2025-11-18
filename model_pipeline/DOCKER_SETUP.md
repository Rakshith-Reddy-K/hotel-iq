# HotelIQ Docker Setup Guide

## 🐳 Quick Start

### 1. First Time Setup

```bash
# Copy environment template
cp .env.example .env

# Edit .env and add your API keys
nano .env  # or use your preferred editor
```

Add your actual keys:
```env
OPENAI_API_KEY=sk-your-actual-key
PINECONE_API_KEY=pcsk-your-actual-key
```

### 2. Build and Run

```bash
# Build images
docker-compose build

# Start services
docker-compose up -d

# View logs
docker-compose logs -f
```

### 3. Access Your Application

- **Chat UI**: http://localhost:8000/chat
- **API Docs**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health
- **Redis**: localhost:6379

---

## 📦 What's Included

### Services

1. **Backend** (FastAPI)
   - Port: 8000
   - Hot reload enabled in development
   - Automatic restarts

2. **Redis** (Caching)
   - Port: 6379
   - Persistent data storage
   - Health checks

3. **Nginx** (Production only)
   - Port: 80/443
   - Reverse proxy
   - Static file serving
   - Rate limiting

---

## 🛠️ Common Commands

### Using Make (Recommended)

```bash
# Show all available commands
make help

# Start development environment
make up

# View logs
make logs

# Open backend shell
make shell

# Restart services
make restart

# Stop everything
make down

# Clean up (remove containers, volumes)
make clean
```

### Using Docker Compose Directly

```bash
# Start services
docker-compose up -d

# Stop services
docker-compose down

# View logs
docker-compose logs -f backend

# Rebuild
docker-compose build --no-cache

# Execute command in container
docker-compose exec backend python -c "print('Hello')"
```

---

## 🔧 Development Mode vs Production Mode

### Development (default)

```bash
# Start with hot reload
make up
# or
docker-compose up
```

Features:
- Code changes automatically reload
- Debug mode enabled
- Source code mounted as volumes
- Single worker process

### Production

```bash
# Start production mode
make up-prod
# or
docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d
```

Features:
- Multiple workers (Gunicorn)
- No code mounting
- Resource limits
- Nginx reverse proxy
- Optimized for performance

---

## 📁 Volume Mounts

### Development

```yaml
volumes:
  - ./backend:/app                    # Backend code (hot reload)
  - ./frontend:/app/frontend          # Frontend files
  - ./booking_requests.json:/app/...  # Booking data
  - hoteliq-data:/app/data/persistent # Persistent storage
  - hoteliq-logs:/app/logs            # Log files
```

### What Gets Persisted

- ✅ Booking requests (`booking_requests.json`)
- ✅ Application logs (`logs/`)
- ✅ Redis data
- ✅ Uploaded files (if any)

---

## 🔍 Troubleshooting

### Container Won't Start

```bash
# Check logs
docker-compose logs backend

# Check if port is already in use
lsof -i :8000

# Rebuild from scratch
make rebuild
```

### Permission Errors

```bash
# Fix permissions on mounted volumes
sudo chown -R $USER:$USER ./backend ./frontend

# Or run as root (not recommended)
docker-compose exec -u root backend bash
```

### API Keys Not Working

```bash
# Verify .env file exists
ls -la .env

# Check environment variables in container
docker-compose exec backend env | grep API_KEY

# Restart services after changing .env
docker-compose down
docker-compose up -d
```

### Redis Connection Issues

```bash
# Check Redis is running
docker-compose ps redis

# Test Redis connection
docker-compose exec redis redis-cli ping

# View Redis logs
docker-compose logs redis
```

### Out of Disk Space

```bash
# Remove unused containers, images, volumes
docker system prune -a

# Remove specific volumes
docker volume rm hoteliq-data hoteliq-logs
```

---

## 🔐 Environment Variables

### Required

```env
OPENAI_API_KEY=sk-...        # OpenAI API key
PINECONE_API_KEY=pcsk-...    # Pinecone API key
```

### Optional

```env
HOTEL_INDEX_NAME=hoteliq-hotels
REVIEWS_INDEX_NAME=hoteliq-reviews
ENVIRONMENT=development
REDIS_URL=redis://redis:6379/0
LOG_LEVEL=INFO
```

---

## 🚀 Deployment

### Local Development

```bash
make up
```

### Production Server

```bash
# Pull latest code
git pull origin main

# Rebuild and deploy
docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d --build

# Check health
curl http://localhost/health
```

### With Nginx (Production)

```bash
# Start with Nginx reverse proxy
docker-compose --profile production up -d
```

---

## 📊 Monitoring

### Health Checks

```bash
# Check all services
make health

# Check specific service
curl http://localhost:8000/health
```

### View Logs

```bash
# All services
make logs

# Backend only
make logs-backend

# Last 100 lines
docker-compose logs --tail=100 backend

# Follow new logs
docker-compose logs -f
```

### Resource Usage

```bash
# Check container stats
docker stats

# Check disk usage
docker system df
```

---

## 🗂️ Backup & Restore

### Backup

```bash
# Backup data using make
make backup-data

# Manual backup
docker-compose exec backend tar czf /tmp/backup.tar.gz \
  /app/booking_requests.json \
  /app/logs
docker cp hoteliq-backend:/tmp/backup.tar.gz ./backup.tar.gz
```

### Restore

```bash
# Copy backup into container
docker cp backup.tar.gz hoteliq-backend:/tmp/

# Extract
docker-compose exec backend tar xzf /tmp/backup.tar.gz -C /
```

---

## 🧪 Testing

### Run Tests in Container

```bash
# Run all tests
make test

# Run specific test file
docker-compose exec backend pytest tests/test_agents.py

# Run with coverage
docker-compose exec backend pytest --cov=agents
```

---

## 📝 File Structure

```
Model_Development_2/
├── backend/
│   ├── Dockerfile              # Backend container definition
│   ├── requirements.txt        # Python dependencies
│   ├── main.py                 # FastAPI application
│   └── agents/                 # Agent modules
├── frontend/
│   └── chat.html              # Chat interface
├── nginx/
│   └── nginx.conf             # Nginx configuration
├── docker-compose.yml         # Development configuration
├── docker-compose.prod.yml    # Production overrides
├── .dockerignore              # Files to exclude from image
├── .env.example               # Environment template
├── .env                       # Your actual environment (git-ignored)
├── Makefile                   # Convenient commands
└── DOCKER_SETUP.md           # This file
```

---

## 🔄 CI/CD Integration

### GitHub Actions Example

```yaml
name: Build and Deploy

on:
  push:
    branches: [main]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Build images
        run: docker-compose build
      
      - name: Run tests
        run: docker-compose run backend pytest
      
      - name: Deploy
        run: docker-compose up -d
```

---

## 💡 Tips & Best Practices

### Performance

- ✅ Use production mode for deployment
- ✅ Enable Redis caching
- ✅ Set resource limits
- ✅ Use multi-stage builds for smaller images

### Security

- ✅ Never commit `.env` file
- ✅ Use secrets management in production
- ✅ Keep images updated
- ✅ Scan for vulnerabilities: `docker scan hoteliq-backend`

### Development

- ✅ Use `make` commands for consistency
- ✅ Check logs regularly
- ✅ Clean up unused resources
- ✅ Use `.dockerignore` to exclude unnecessary files

---

## 🆘 Getting Help

### Debug Mode

```bash
# Start with debug output
docker-compose up --verbose

# Run container interactively
docker-compose run --rm backend /bin/bash
```

### Common Issues

| Issue | Solution |
|-------|----------|
| Port already in use | Change port in docker-compose.yml or stop conflicting service |
| Container keeps restarting | Check logs: `docker-compose logs backend` |
| Changes not reflecting | Rebuild: `make rebuild` |
| Out of memory | Increase Docker memory limit in Docker Desktop |

---

## 📚 Additional Resources

- [Docker Compose Documentation](https://docs.docker.com/compose/)
- [FastAPI Docker Guide](https://fastapi.tiangolo.com/deployment/docker/)
- [Docker Best Practices](https://docs.docker.com/develop/dev-best-practices/)

---

## ✨ Next Steps

1. ✅ Set up environment variables
2. ✅ Start development server: `make up`
3. ✅ Access chat UI: http://localhost:8000/chat
4. ✅ Monitor logs: `make logs`
5. ✅ Run tests: `make test`
6. 🚀 Deploy to production!

Happy coding! 🎉

