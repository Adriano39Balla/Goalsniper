# -----------------------------------------------------
# GOALSNIPER-AI DOCKERFILE (Railway Compatible)
# -----------------------------------------------------

FROM python:3.10-slim

WORKDIR /app

# -----------------------------------------------------
# Install system dependencies (Railway Safe)
# -----------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    gfortran \
    git \
    && rm -rf /var/lib/apt/lists/*

# -----------------------------------------------------
# Install Python dependencies
# -----------------------------------------------------
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# -----------------------------------------------------
# Copy project files
# -----------------------------------------------------
COPY . .

EXPOSE 8000

# -----------------------------------------------------
# Start ML prediction engine
# -----------------------------------------------------
CMD ["python", "main.py"]
