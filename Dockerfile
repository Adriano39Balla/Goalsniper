# --- Dockerfile ---
FROM python:3.11-slim

# Safe defaults
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Runtime deps for psycopg2-binary
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install deps first (better cache)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app
COPY . .

# Railway injects PORT; default 8080
ENV PORT=8080

# Start the app (1 worker so the scheduler isn’t duplicated)
CMD ["gunicorn", "-w", "1", "-k", "gthread", "-t", "120", "main:app"]
