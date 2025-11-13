# -----------------------------------------------------
# GOALSNIPER-AI DOCKERFILE (FINAL PRODUCTION VERSION)
# -----------------------------------------------------

FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# -----------------------------------------------------
# Install system dependencies for numpy / pandas / sklearn
# -----------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    gfortran \
    libopenblas-dev \
    liblapack-dev \
    libblas-dev \
    libgfortran5 \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# -----------------------------------------------------
# Copy dependencies and install Python packages
# -----------------------------------------------------
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# -----------------------------------------------------
# Copy application
# -----------------------------------------------------
COPY . .

# Ensure model directory always exists
RUN mkdir -p /app/models

# -----------------------------------------------------
# Expose port for Railway
# -----------------------------------------------------
EXPOSE 8000

# -----------------------------------------------------
# HEALTHCHECK for Railway
# -----------------------------------------------------
HEALTHCHECK --interval=20s --timeout=5s --start-period=15s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

# -----------------------------------------------------
# START COMMAND
# -----------------------------------------------------
CMD ["python", "main.py"]
