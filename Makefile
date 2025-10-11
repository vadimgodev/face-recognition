.PHONY: help up down restart logs ps clean db-reset test enroll recognize list-faces delete-all health

# Default target
help:
	@echo "Face Recognition API - Available Commands"
	@echo ""
	@echo "Docker Management:"
	@echo "  make up              - Start all containers"
	@echo "  make down            - Stop all containers"
	@echo "  make restart         - Restart all containers"
	@echo "  make logs            - View all logs (follow mode)"
	@echo "  make logs-app        - View app logs only"
	@echo "  make ps              - Show running containers"
	@echo "  make clean           - Stop and remove all containers, volumes"
	@echo ""
	@echo "Database:"
	@echo "  make db-reset        - Reset database (delete all data)"
	@echo "  make db-shell        - Connect to PostgreSQL shell"
	@echo "  make db-migrate      - Run database migrations"
	@echo ""
	@echo "Testing:"
	@echo "  make test            - Run all tests"
	@echo "  make health          - Check API health"
	@echo "  make list-faces      - List all enrolled faces"
	@echo "  make delete-all      - Delete all faces from database"
	@echo ""
	@echo "Face Operations:"
	@echo "  make enroll FILE=path/to/photo.jpg NAME=\"User Name\"  - Enroll a face"
	@echo "  make recognize FILE=path/to/photo.jpg                  - Recognize a face"
	@echo ""
	@echo "Quality & Performance:"
	@echo "  make stats           - Show quality statistics"
	@echo "  make check-quality FILE=path/to/photo.jpg             - Check photo quality"
	@echo ""
	@echo "Maintenance:"
	@echo "  make fix-models      - Fix InsightFace model directory structure (run after restart)"

# Docker commands
up:
	@echo "Starting all containers..."
	docker-compose up -d
	@echo "Waiting for services to be ready..."
	@sleep 5
	@echo "Services started! API: http://localhost:8000/docs"

down:
	@echo "Stopping all containers..."
	docker-compose down

restart:
	@echo "Restarting all containers..."
	docker-compose down
	docker-compose up -d
	@echo "Waiting for services to be ready..."
	@sleep 5
	@echo "Services restarted!"

logs:
	docker-compose logs -f

logs-app:
	docker-compose logs -f app

ps:
	docker-compose ps

clean:
	@echo "WARNING: This will remove all containers, volumes, and data!"
	@echo "Press Ctrl+C to cancel, or wait 5 seconds to continue..."
	@sleep 5
	docker-compose down -v
	@echo "Cleanup complete!"

# Database commands
db-reset:
	@echo "Resetting database..."
	docker-compose exec postgres psql -U postgres -d facedb -c "TRUNCATE TABLE faces CASCADE;"
	@echo "Database reset complete!"

db-shell:
	docker-compose exec postgres psql -U postgres -d facedb

db-migrate:
	docker-compose exec app alembic upgrade head

# Testing commands
test:
	docker-compose exec app pytest

health:
	@echo "Checking API health..."
	@curl -s http://localhost:8000/health | jq . || echo "API not responding"

list-faces:
	@echo "Enrolled faces:"
	@curl -s "http://localhost:8000/api/v1/faces?limit=100" | jq '.faces[] | {id, user_name, photo_type, quality_score, created_at}'

delete-all:
	@echo "Deleting all faces..."
	@curl -s "http://localhost:8000/api/v1/faces?limit=100" | jq -r '.faces[].id' | xargs -I {} curl -s -X DELETE "http://localhost:8000/api/v1/faces/{}" | jq -s 'map(select(.success == true)) | length' | xargs -I {} echo "{} faces deleted"
	@echo "All faces deleted!"

# Face operations
enroll:
	@if [ -z "$(FILE)" ] || [ -z "$(NAME)" ]; then \
		echo "Usage: make enroll FILE=path/to/photo.jpg NAME=\"User Name\""; \
		echo "Example: make enroll FILE=/tmp/photo.jpg NAME=\"John Doe\""; \
		exit 1; \
	fi
	@echo "Enrolling face: $(NAME)"
	@curl -s -X POST http://localhost:8000/api/v1/faces/enroll \
		-F "image=@$(FILE)" \
		-F "user_name=$(NAME)" \
		-F "user_email=$(NAME)@example.com" | jq .

recognize:
	@if [ -z "$(FILE)" ]; then \
		echo "Usage: make recognize FILE=path/to/photo.jpg"; \
		echo "Example: make recognize FILE=/tmp/vadim.jpg"; \
		exit 1; \
	fi
	@echo "Recognizing face..."
	@curl -s -X POST http://localhost:8000/api/v1/faces/recognize \
		-F "image=@$(FILE)" \
		-F "confidence_threshold=0.6" | jq .

# Quality & Performance
stats:
	@echo "Quality Statistics:"
	@docker-compose exec postgres psql -U postgres -d facedb -c "\
		SELECT \
			photo_type, \
			COUNT(*) as count, \
			ROUND(AVG(quality_score)::numeric, 3) as avg_quality, \
			ROUND(MIN(quality_score)::numeric, 3) as min_quality, \
			ROUND(MAX(quality_score)::numeric, 3) as max_quality \
		FROM faces \
		WHERE quality_score IS NOT NULL \
		GROUP BY photo_type;"

check-quality:
	@if [ -z "$(FILE)" ]; then \
		echo "Usage: make check-quality FILE=path/to/photo.jpg"; \
		echo "Example: make check-quality FILE=/tmp/vadim.jpg"; \
		exit 1; \
	fi
	@echo "Checking photo quality..."
	@curl -s -X POST http://localhost:8000/api/v1/faces/enroll \
		-F "image=@$(FILE)" \
		-F "user_name=Quality Test" \
		-F "user_email=test@example.com" | jq 'if .success then {quality_score: .face.quality_score, message: "✅ Quality passed"} else {error: .detail} end'
	@echo "Cleaning up test enrollment..."
	@curl -s "http://localhost:8000/api/v1/faces?limit=1" | jq -r '.faces[] | select(.user_name == "Quality Test") | .id' | xargs -I {} curl -s -X DELETE "http://localhost:8000/api/v1/faces/{}" > /dev/null

# Development commands
dev-setup:
	@echo "Setting up development environment..."
	cp .env.example .env
	@echo "Created .env file. Please edit it with your settings."
	@echo "Run 'make up' to start the services."

fix-models:
	@echo "Fixing InsightFace model directory structure..."
	docker-compose exec app bash /app/scripts/fix-insightface-models.sh
	@echo "Restarting app to reload models..."
	docker-compose restart app
	@echo "✅ Models fixed and app restarted"

shell-app:
	docker-compose exec app bash

shell-db:
	docker-compose exec postgres bash

# Monitoring
monitor:
	@echo "Monitoring logs (Ctrl+C to exit)..."
	docker-compose logs -f app | grep -E "(INFO|ERROR|WARNING)"

errors:
	@echo "Recent errors:"
	docker-compose logs app | grep ERROR | tail -20
