FROM python:3.10-slim

WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . /app/

# Make sure assets directory exists with proper permissions
RUN mkdir -p /app/assets/static && chmod -R 755 /app/assets

# Create necessary directory for HuggingFace Spaces
RUN mkdir -p /app/.huggingface

# Expose port
EXPOSE 8000

# Environment variables for HuggingFace Spaces
ENV GRADIO_ALLOW_FLAGGING=never \
    GRADIO_SERVER_NAME=0.0.0.0 \
    GRADIO_SERVER_PORT=8000

# Command to run the application
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"] 