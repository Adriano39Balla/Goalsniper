# -----------------------------------------------------
# GOALSNIPER-AI DOCKERFILE (Final + Stable for Railway)
# -----------------------------------------------------

FROM python:3.10-slim

WORKDIR /app

# -----------------------------------------------------
# Install system libraries (Aesara + PyMC Safe)
# -----------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    gfortran \
    libopenblas-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

# -----------------------------------------------------
# Install Python dependencies
# -----------------------------------------------------
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# -----------------------------------------------------
# Copy project
# -----------------------------------------------------
COPY . .

EXPOSE 8000

CMD ["python", "main.py"]
