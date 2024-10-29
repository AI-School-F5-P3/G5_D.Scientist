# Base image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy frontend requirements
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy frontend code
COPY . .

# Remove unnecessary files and directories
RUN find . -type d -name "__pycache__" -exec rm -r {} + 2>/dev/null || true \
    && find . -type d -name "*.pyc" -exec rm -r {} + 2>/dev/null || true

# Expose the port used by the frontend application
EXPOSE 3000

# Command to run the application (adjust according to your framework)
CMD ["streamlit", "run", "app.py", "--server.port", "3000", "--server.address", "0.0.0.0"]