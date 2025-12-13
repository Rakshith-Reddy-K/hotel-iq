# HotelIQ Docker Commands
# ========================

.PHONY: help build up down restart logs shell test clean

# Default target
.DEFAULT_GOAL := help

help: ## Show this help message
	@echo "HotelIQ Docker Commands:"
	@echo "========================"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

build: ## Build Docker images
	docker-compose build

up: ## Start all services (development)
	docker-compose up -d
	@echo "✅ Services started!"
	@echo "🌐 Backend API: http://localhost:8000"
	@echo "💬 Chat UI: http://localhost:8000/chat"
	@echo "📊 API Docs: http://localhost:8000/docs"

up-prod: ## Start all services (production mode)
	docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d
	@echo "✅ Production services started!"

down: ## Stop all services
	docker-compose down

restart: ## Restart all services
	docker-compose restart

logs: ## View logs (follow mode)
	docker-compose logs -f

logs-backend: ## View backend logs only
	docker-compose logs -f backend

logs-redis: ## View redis logs only
	docker-compose logs -f redis

shell: ## Open shell in backend container
	docker-compose exec backend /bin/bash

shell-redis: ## Open Redis CLI
	docker-compose exec redis redis-cli

ps: ## Show running containers
	docker-compose ps

test: ## Run tests inside container
	docker-compose exec backend pytest

clean: ## Remove containers, volumes, and images
	docker-compose down -v
	docker system prune -f

rebuild: ## Rebuild and restart services
	docker-compose down
	docker-compose build --no-cache
	docker-compose up -d

# Development commands
dev: ## Start development environment with logs
	docker-compose up

prod: ## Start production environment
	docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d

# Database/Data commands
backup-data: ## Backup booking requests and logs
	@mkdir -p backups
	docker-compose exec backend tar czf /tmp/backup.tar.gz /app/booking_requests.json /app/logs
	docker cp hoteliq-backend:/tmp/backup.tar.gz backups/backup-$$(date +%Y%m%d-%H%M%S).tar.gz
	@echo "✅ Backup created in backups/"

# Utility commands
health: ## Check health of services
	@curl -s http://localhost:8000/health || echo "❌ Backend is down"
	@docker-compose exec redis redis-cli ping || echo "❌ Redis is down"

install: ## Copy .env.example to .env (first time setup)
	@if [ ! -f .env ]; then \
		cp .env.example .env; \
		echo "✅ Created .env file. Please edit it with your API keys!"; \
	else \
		echo "⚠️  .env file already exists"; \
	fi

