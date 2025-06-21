# Dockerfile for Insurance Fraud Project
# Builds and prepares the environment for running the Flask web app

FROM python:3.11

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Generate the data and train the model at build time
RUN python claims_fraud/data_generator.py
RUN python claims_fraud/model.py

# Expose the web server port
EXPOSE 8080

# Start the Flask app with Gunicorn (recommended for production)
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "claims_fraud.app:app"]