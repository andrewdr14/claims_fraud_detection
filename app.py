import os
import pandas as pd
import joblib
from flask import Flask, render_template, send_file
from pymongo import MongoClient
from dotenv import load_dotenv
from sklearn.metrics import classification_report
import model  # Import fraud detection model

# Load environment variables
load_dotenv()
mongo_uri = os.getenv("MONGO_URI")
client = MongoClient(mongo_uri)

app = Flask(__name__)

# Load pre-trained fraud detection model
if os.path.exists("fraud_model.pkl"):
    model.model = joblib.load("fraud_model.pkl")
    print("✅ Pre-trained model loaded successfully!")

@app.route("/")
def evaluation():
    """Run fraud model evaluation and display results."""
    df = model.fetch_data()  # Load claims data

    df["fraud_reported"] = df["fraud_reported"].astype(str).str.strip().str.capitalize()
    label_map = {"Yes": 1, "No": 0}
    df["fraud_label"] = df["fraud_reported"].map(label_map)

    y_true = df["fraud_label"]
    y_pred = model.model.predict(df[model.trained_features])

    # Generate classification report dynamically
    evaluation_results = classification_report(y_true, y_pred, output_dict=True)

    # Generate summary statistics
    summary_stats = df.describe().to_dict()

    return render_template(
        "results.html",
        features=model.trained_features,
        evaluation_results=evaluation_results,
        summary_stats=summary_stats
    )

@app.route("/download-data")
def download_data():
    """Allow users to download the dataset."""
    return send_file("motor_insurance_claims.csv", as_attachment=True)

if __name__ == "__main__":
    # Dynamically assign Render's port for deployment
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port, debug=False)