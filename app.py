import os
import io
import pandas as pd
from flask import Flask, request, render_template, send_file, redirect, url_for
from werkzeug.utils import secure_filename
import model
from pymongo import MongoClient
from dotenv import load_dotenv

# Load environment + MongoDB setup
load_dotenv()
mongo_uri = os.getenv("MONGO_URI")
client = MongoClient(mongo_uri)
db = client["claims-fraud-db"]

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
OUTPUT_FOLDER = "outputs"
ALLOWED_EXTENSIONS = {"csv"}
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["OUTPUT_FOLDER"] = OUTPUT_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def get_versioned_collection_name(base):
    """Create a versioned collection name if one already exists."""
    existing = db.list_collection_names()
    if base not in existing:
        return base
    i = 2
    while f"{base}_{i}" in existing:
        i += 1
    return f"{base}_{i}"

@app.route("/", methods=["GET", "POST"])
def upload_file():
    if request.method == "POST":
        file = request.files["file"]
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            base = os.path.splitext(filename)[0]
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(filepath)

            # Load, enrich, score
            df = pd.read_csv(filepath)
            model.initialize_model(df)
            enriched_df = model.predict_fraud(df)

            # Get unique collection name and store in MongoDB
            collection_name = get_versioned_collection_name(base)
            db[collection_name].insert_many(enriched_df.to_dict(orient="records"))

            # Save enriched CSV locally for download
            output_path = os.path.join(app.config["OUTPUT_FOLDER"], f"{collection_name}.csv")
            enriched_df.to_csv(output_path, index=False)

            return redirect(url_for("results", filename=f"{collection_name}.csv"))

    return render_template("index.html")

@app.route("/results")
def results():
    filename = request.args.get("filename")
    return render_template("results.html", filename=filename)

@app.route("/download/<filename>")
def download_file(filename):
    path = os.path.join(app.config["OUTPUT_FOLDER"], filename)
    return send_file(path, as_attachment=True)

@app.route("/collections")
def list_collections():
    return {"collections": db.list_collection_names()}

@app.route("/predictions/<collection_name>")
def get_predictions(collection_name):
    return list(db[collection_name].find({}, {"_id": 0}))

if __name__ == "__main__":
    app.run(debug=False)