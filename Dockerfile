# Use a lightweight Python image
FROM python:3.11-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install Playwright and automatically install the required system dependencies for Chromium
RUN playwright install chromium --with-deps

# Copy the rest of your application code
COPY . .

# Railway automatically assigns a PORT environment variable. 
# We set a default of 8001 just in case, but Uvicorn will use Railway's port.
ENV PORT=8001
EXPOSE $PORT

# Start the FastAPI application
CMD uvicorn main:app --host 0.0.0.0 --port $PORT