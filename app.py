import os
import io
import pandas as pd
from flask import Flask, request, render_template, send_file, redirect, url_for
from werkzeug.utils import secure_filename
import model
from pymongo import MongoClient
from dotenv import load_dotenv

# Environment setup
load_dotenv()
mongo_uri = os.getenv("MONGO_URI")
client = MongoClient(mongo_uri)
db = client["claims-fraud-db"]

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
TEMPLATE_FOLDER = "static/template_output"
OUTPUT_FOLDER = "outputs"
ALLOWED_EXTENSIONS = {"csv", "xlsx"}

for folder in [UPLOAD_FOLDER, TEMPLATE_FOLDER, OUTPUT_FOLDER]:
    os.makedirs(folder, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# 🎯 Core feature list
FULL_FEATURES = [
    'age', 'months_as_customer', 'policy_deductable', 'policy_annual_premium',
    'umbrella_limit', 'capital_gains', 'capital_loss', 'incident_hour_of_the_day',
    'number_of_vehicles_involved', 'bodily_injuries', 'witnesses',
    'total_claim_amount', 'injury_claim', 'property_claim', 'vehicle_claim'
]

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def get_versioned_collection_name(base):
    existing = db.list_collection_names()
    if base not in existing:
        return base
    i = 2
    while f"{base}_{i}" in existing:
        i += 1
    return f"{base}_{i}"

@app.route("/", methods=["GET"])
def index():
    return render_template("feature_select.html", features=FULL_FEATURES)

@app.route("/generate-template", methods=["POST"])
def generate_template():
    selected = request.form.getlist("features")
    if not selected:
        return "Please select at least one feature", 400
    df = pd.DataFrame(columns=selected)  # Include fraud_reported for training
    template_name = "custom_template.xlsx"
    template_path = os.path.join(TEMPLATE_FOLDER, template_name)
    df.to_excel(template_path, index=False)
    return send_file(template_path, as_attachment=True)

@app.route("/upload", methods=["POST"])
def upload():
    file = request.files["file"]
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        base = os.path.splitext(filename)[0]
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)

        # Detect file type and read
        ext = filename.rsplit(".", 1)[1].lower()
        if ext == "csv":
            df = pd.read_csv(filepath)
        else:
            df = pd.read_excel(filepath)

        model.initialize_model(df)
        enriched = model.predict_fraud(df)

        collection_name = get_versioned_collection_name(base)
        db[collection_name].insert_many(enriched.to_dict(orient="records"))

        output_file = f"{collection_name}.csv"
        output_path = os.path.join(OUTPUT_FOLDER, output_file)
        enriched.to_csv(output_path, index=False)

        return redirect(url_for("results", filename=output_file))

    return "Invalid file", 400

@app.route("/results")
def results():
    filename = request.args.get("filename")
    return render_template("results.html", filename=filename)

@app.route("/download/<filename>")
def download_file(filename):
    return send_file(os.path.join(OUTPUT_FOLDER, filename), as_attachment=True)

if __name__ == "__main__":
    app.run(debug=False)