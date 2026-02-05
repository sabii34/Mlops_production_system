FROM python:3.10-slim

# Avoid .pyc and enable unbuffered logs
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install dependencies first (better caching)
COPY requirements-docker.txt /app/requirements-docker.txt
RUN pip install --no-cache-dir --default-timeout=300 --retries 10 -r requirements-docker.txt



# Copy application code + required artifacts
COPY src /app/src
COPY app.py /app/app.py
COPY models /app/models

# Expose API port
EXPOSE 8000

# Run the API
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
