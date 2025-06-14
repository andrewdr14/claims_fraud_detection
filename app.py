import os
import pandas as pd
import numpy as np
from flask import Flask, request, render_template, jsonify
from werkzeug.utils import secure_filename
import model  # Import the entire module
from pymongo import MongoClient
from dotenv import load_dotenv

# 1️⃣ Load Environment Variables
load_dotenv()

# 2️⃣ Get MongoDB URI from Environment Variables
mongo_uri = os.getenv("MONGO_URI")  # Ensure this is correctly set in Render

# 3️⃣ Connect to MongoDB Atlas
client = MongoClient(mongo_uri)
db = client["claims-fraud-db"]  # Use the correct database name
collection = db["insurance_claims"]  # Use the correct collection name

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
ALLOWED_EXTENSIONS = {"csv"}

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Ensure upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    """Check if the uploaded file is a CSV."""
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/", methods=["GET", "POST"])
def upload_file():
    """Handle file upload and fraud prediction."""
    if request.method == "POST":
        file = request.files["file"]
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(filepath)

            # Process dataset
            df = pd.read_csv(filepath)
            fraud_probs = model.predict_fraud(df)  # Get fraud probabilities

            # Add probabilities to dataset
            df["Fraud Probability"] = fraud_probs

            # Store predictions in MongoDB
            collection.insert_many(df.to_dict(orient="records"))

            return jsonify({"message": "Predictions stored in MongoDB", "data": df.to_dict(orient="records")})

    return render_template("index.html")

@app.route("/predictions", methods=["GET"])
def get_predictions():
    """Retrieve stored fraud predictions from MongoDB."""
    predictions = list(collection.find({}, {"_id": 0}))  # Exclude MongoDB's default `_id` field
    return jsonify(predictions)

if __name__ == "__main__":
    import sys
    if "flask" not in sys.argv[0]:  # Prevent duplicate execution
        app.run(debug=False)