# Local dev image (docker-compose). Installs the full requirements.txt,
# including test deps, which the slim Dockerfile.web deliberately omits.
# Keep the base in sync with Dockerfile.web/.worker/.beat and CI (3.11).
FROM python:3.14-slim-bookworm

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Create and set work directory
WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project
COPY . .

# NOTE: local dev image only. Production uses Dockerfile.web + Dockerfile.worker.
# migrate + collectstatic moved into CMD so build-time doesn't require a DB.
CMD ["sh", "-c", "python manage.py migrate --noinput && python manage.py collectstatic --noinput && gunicorn --bind 0.0.0.0:8000 querygrade.wsgi"]
