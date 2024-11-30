# Base image with FastAPI and Uvicorn
FROM tiangolo/uvicorn-gunicorn-fastapi:python3.9

# Set the working directory inside the container
WORKDIR /app

# Copy only the requirements file first to leverage Docker's caching
COPY requirements.txt /app/requirements.txt

# Install dependencies
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy the rest of the application code
COPY . .

# Expose the application port
EXPOSE 80

# Command to run the application
CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "80"]
