# RESPONSIBILITY: Dockerfile (Backend)
# --------------------------------------
# Builds the Docker image for the FastAPI backend service.
# - Uses Python 3.11 slim as the base image
# - Copies source code and installs dependencies
# - Exposes port 8000 (internal Docker network only)
# - Starts FastAPI with uvicorn on container startup
# - Used by docker-compose.yml as the 'backend' service

FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]

