import os
import pandas as pd
from flask import Flask, request, render_template, jsonify
from werkzeug.utils import secure_filename
import model
from pymongo import MongoClient
from dotenv import load_dotenv

# Load environment & connect to Mongo
load_dotenv()
mongo_uri = os.getenv("MONGO_URI")
client = MongoClient(mongo_uri)
db = client["claims-fraud-db"]
collection = db["insurance_claims"]

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

            df = pd.read_csv(filepath)
            enriched_df = model.predict_fraud(df)

            if not enriched_df.empty:
                collection.insert_many(enriched_df.to_dict(orient="records"))

            return jsonify({
                "message": "Predictions stored in MongoDB",
                "data": enriched_df.to_dict(orient="records")
            })

    return render_template("index.html")

@app.route("/predictions", methods=["GET"])
def get_predictions():
    return jsonify(list(collection.find({}, {"_id": 0})))

if __name__ == "__main__":
    app.run(debug=False)