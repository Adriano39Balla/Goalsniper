# -----------------------------------------------------
# GOALSNIPER-AI DOCKERFILE
# -----------------------------------------------------

FROM python:3.10-slim

# Set working directory
WORKDIR /app

# -----------------------------------------------------
# Install system dependencies for numpy, pandas, pymc
# -----------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    libatlas-base-dev \
    liblapack-dev \
    libblas-dev \
    libgfortran5 \
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

# Expose port for Flask (main.py binds to 8000)
EXPOSE 8000

# -----------------------------------------------------
# Start the prediction engine
# -----------------------------------------------------
CMD ["python", "main.py"]
