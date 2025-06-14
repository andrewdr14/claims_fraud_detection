import os
import pandas as pd
import numpy as np
from flask import Flask, request, render_template, send_file
from werkzeug.utils import secure_filename
import model  # Import the entire module

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
            fraud_probs = model.predict_fraud(df)  # Corrected function call

            # Add probabilities to dataset
            df["Fraud Probability"] = fraud_probs

            # Save processed file
            output_filepath = os.path.join(app.config["UPLOAD_FOLDER"], "processed_" + filename)
            df.to_csv(output_filepath, index=False)

            return send_file(output_filepath, as_attachment=True)

    return render_template("index.html")

if __name__ == "__main__":
    import sys
    if "flask" not in sys.argv[0]:  # Prevent duplicate execution
        app.run(debug=False)