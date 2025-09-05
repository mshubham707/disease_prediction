# Use official lightweight Python image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Copy files
COPY . /app

# Install dependencies
RUN pip install --upgrade pip \
 && pip install --no-cache-dir -r requirements.tx

# Expose port 7860 (HF default)
EXPOSE 7860

# Run Flask app
CMD ["python", "src/app.py"]
