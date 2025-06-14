from flask import Flask, request, render_template, jsonify
import pandas as pd
import numpy as np
import pymysql
from sqlalchemy import create_engine
from dotenv import load_dotenv
import os
from model import predict_fraud

# Load environment variables
load_dotenv()

# Database connection using .env variables
db_config = {
    'user': os.getenv('DB_USER'),
    'password': os.getenv('DB_PASSWORD'),
    'host': os.getenv('DB_HOST'),
    'database': os.getenv('DB_NAME')
}

# Create database engine
engine = create_engine(f"mysql+pymysql://{db_config['user']}:{db_config['password']}@{db_config['host']}/{db_config['database']}")

# Flask app setup
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json if request.is_json else request.form
    features = np.array([[data.get('age'), data.get('months_as_customer'), data.get('policy_deductable'),
                          data.get('policy_annual_premium'), data.get('umbrella_limit'), data.get('capital_gains'),
                          data.get('capital_loss'), data.get('incident_hour_of_the_day'), data.get('number_of_vehicles_involved'),
                          data.get('bodily_injuries'), data.get('witnesses'), data.get('total_claim_amount'),
                          data.get('injury_claim'), data.get('property_claim'), data.get('vehicle_claim')]])

    prediction = predict_fraud(features)
    return jsonify({'fraud_detected': bool(prediction)})

if __name__ == '__main__':
    app.run(debug=True)