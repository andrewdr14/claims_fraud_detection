import os
import pandas as pd
from flask import Flask, request, render_template, jsonify
from werkzeug.utils import secure_filename
import model
from pymongo import MongoClient
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
mongo_uri = os.getenv("MONGO_URI")
client = MongoClient(mongo_uri)
db = client["claims-fraud-db"]

# Flask config
app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
ALLOWED_EXTENSIONS = {"csv"}
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/", methods=["GET", "POST"])
def upload_file():
    if request.method == "POST":
        file = request.files["file"]
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(filepath)

            # Load uploaded file
            df = pd.read_csv(filepath)

            # Train model (only once)
            model.initialize_model(df)

            # Get fraud probabilities
            enriched_df = model.predict_fraud(df)

            # Store in a collection named after the file (excluding extension)
            collection_name = os.path.splitext(filename)[0]
            file_collection = db[collection_name]
            file_collection.insert_many(enriched_df.to_dict(orient="records"))

            return jsonify({
                "message": f"Stored in MongoDB collection '{collection_name}'",
                "data": enriched_df.to_dict(orient="records")
            })

    return render_template("index.html")

@app.route("/collections", methods=["GET"])
def list_collections():
    return jsonify({"collections": db.list_collection_names()})

@app.route("/predictions/<collection_name>", methods=["GET"])
def get_predictions(collection_name):
    collection = db[collection_name]
    return jsonify(list(collection.find({}, {"_id": 0})))

if __name__ == "__main__":
    app.run(debug=False)