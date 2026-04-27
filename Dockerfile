# Use the official lightweight Python image.
FROM python:3.11-slim

# Allow statements and log messages to immediately appear in the Knative logs
ENV PYTHONUNBUFFERED True

# Copy local code to the container image.
ENV APP_HOME /app
WORKDIR $APP_HOME
COPY . ./

# Install production dependencies.
RUN pip install --no-cache-dir -r requirements.txt

# Disable training on startup for fast container boot (prevents port timeout)
ENV TRAIN_MODELS_ON_STARTUP false

# Run the web service on container startup using Uvicorn.
# Using standard host and dynamic port assignment for PaaS providers
CMD exec uvicorn src.main:app --host 0.0.0.0 --port ${PORT:-10000}
