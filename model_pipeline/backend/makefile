IMAGE_NAME = hoteliq-backend
CONTAINER_NAME = hoteliq
PORT = 8000

.PHONY: build run stop logs clean

# Build Docker image
build:
	docker build -t $(IMAGE_NAME) .

# Run container
run:
	docker run -d -p $(PORT):$(PORT) --name $(CONTAINER_NAME) --env-file .env $(IMAGE_NAME)
	@echo "✓ Running at http://localhost:$(PORT)"

# Stop container
stop:
	docker stop $(CONTAINER_NAME)

# View logs
logs:
	docker logs -f $(CONTAINER_NAME)

# Clean up (stop and remove)
clean:
	docker stop $(CONTAINER_NAME) || true
	docker rm $(CONTAINER_NAME) || true

# Rebuild (clean, build, run)
rebuild: clean build run