# Start with a lightweight Python image
FROM python:3.11-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the dependency list before copying application code
COPY requirements.txt ./

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code and source
COPY main.py train.py ./
COPY src ./src
COPY backend ./backend
COPY scripts ./scripts

# Copy the canonical model artifact and the empty legacy compatibility directory.
# Model is loaded from MLflow Model Registry at runtime, not copied into image

# Document the port used by FastAPI application
EXPOSE 8000

# Start the API server when the container launches
ENV MLFLOW_TRACKING_URI=http://host.docker.internal:7001
EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]