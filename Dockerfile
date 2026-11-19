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
COPY models ./models
COPY artifacts ./artifacts

# Document the port used by FastAPI application
EXPOSE 8000

# Start the API server when the container launches
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
