# Claims Fraud Detection

A data-driven web application that detects and analyzes insurance claim fraud using machine learning models (Random Forest, XGBoost) and MongoDB for data storage. The project provides a Flask-based web interface to evaluate model performance and review fraud statistics.

---

## Features

- **Synthetic Data Generation:** Create realistic motor insurance claim datasets with Faker.
- **Data Storage:** Store and retrieve claim data from MongoDB Atlas.
- **Model Training:** Train RandomForest and XGBoost classifiers to detect fraud.
- **Web Dashboard:** Visualize classification metrics and summary statistics via Flask.
- **Downloadable Data:** Users can download the generated dataset from the web UI.

---

## Prerequisites

- Python 3.7+
- MongoDB Atlas account (free tier is sufficient)
- (Optional) Virtual environment tool (like `venv`)

---

## MongoDB Atlas Setup

This project stores all claim data in a MongoDB Atlas database.  
**You must set up MongoDB Atlas before running the code.**

1. **Sign up / Log in:** Go to [mongodb.com/cloud/atlas](https://www.mongodb.com/cloud/atlas).
2. **Create a Cluster:** Deploy a new free cluster.
3. **Add Your IP Address:** In "Network Access", add your current IP address.
4. **Create a Database User:** In "Database Access", add a user with read/write privileges.
5. **Get Your Connection String:**
    - In Atlas, click "Connect" > "Connect your application".
    - Copy the connection string (e.g. `mongodb+srv://<username>:<password>@cluster0.xxxxx.mongodb.net/`)
6. **Set Up the `.env` File:**
    - In your project root directory, create a file named `.env`.
    - Add the following line, replacing with your credentials:
      ```
      MONGO_URI=mongodb+srv://<username>:<password>@cluster0.xxxxx.mongodb.net/claims-fraud-db?retryWrites=true&w=majority&appName=<your-app-name>
      ```
    - **Note:** Do not wrap the URI in quotes.  
    - The database (`claims-fraud-db`) and collection (`motor_insurance_claims`) will be created automatically.

---

## Installation

1. **Clone the repository:**

    ```bash
    git clone https://github.com/andrewdr14/claims-fraud-detection.git
    cd claims-fraud-detection
    ```

2. **Set up Python environment:**

    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    pip install -r requirements.txt
    ```

3. **Configure environment variables:**
    - Follow the steps above to create and fill in the `.env` file.

---

## Project Structure & Running as a Package

This project is now organized as a Python package for better maintainability and scalability.

### Directory Structure

```
claims-fraud-detection/
├── claims_fraud/
│   ├── __init__.py
│   ├── data_generator.py
│   ├── model.py
│   └── app.py
├── requirements.txt
├── README.md
├── .env
└── ...
```

### How to Run Each Script

All scripts should be run as **modules** from the project root using the `-m` flag.  
This ensures relative imports work correctly and keeps your environment clean.

**Step-by-step:**

1. **Generate Data and Upload to MongoDB**

    ```bash
    python -m claims_fraud.data_generator
    ```

2. **Train the Models**

    ```bash
    python -m claims_fraud.model
    ```

3. **Launch the Web Application**

    ```bash
    python -m claims_fraud.app
    ```

The web application will be accessible at [http://localhost:8080](http://localhost:8080).

---

### Notes

- **Do not run the scripts from inside the `claims_fraud/` directory.**  
  Always run from the root (the directory containing `claims_fraud/`).
- **Make sure your `.env` file is in the project root.**
- **All inter-module imports within `claims_fraud/` use relative imports** (e.g., `from . import model`).

---

**Why this structure?**  
Organizing your code as a package makes it easier to maintain, test, and extend, and is considered best practice for any project beyond a single script.

---

## Usage: Order of Execution

**You must run the programs in this order:**

1. ### 1. Generate and Upload Data

    ```bash
    python -m claims_fraud.data_generator
    ```

    - This will create synthetic claim data, save it as `motor_insurance_claims.csv`, and upload it to your MongoDB Atlas cluster.
    - If MongoDB is not set up correctly (see above), this step will fail.

2. ### 2. Train the Models

    ```bash
    python -m claims_fraud.model
    ```

    - This will fetch data from MongoDB, train the RandomForest and XGBoost models, and save them as `random_forest.pkl` and `xgboost.pkl`.

3. ### 3. Run the Web Application

    ```bash
    python -m claims_fraud.app
    ```

    - The app runs a Flask server. Open your browser and go to [http://localhost:8080](http://localhost:8080).
    - Here you can view model performance and download the dataset.

---

## Project Structure

```
├── claims_fraud/
│   ├── __init__.py
│   ├── app.py                  # Flask web app for evaluation and reporting
│   ├── data_generator.py       # Synthetic data generator and MongoDB uploader
│   ├── model.py                # Model training, saving/loading, and database logic
├── requirements.txt            # Python dependencies
├── .env                        # Environment variables (do not commit this file)
├── motor_insurance_claims.csv  # Example generated dataset
├── templates/
│   └── results.html            # HTML template for results display
└── README.md
```

---

## Troubleshooting

- **Cannot connect to MongoDB Atlas:**  
  - Check that your `.env` file is present and correct.
  - Ensure your IP is whitelisted in Atlas.
  - Make sure Atlas user credentials are correct.
  - The connection string should start with `mongodb+srv://`.

- **Order matters:**  
  - If you run `model.py` or `app.py` before `data_generator.py`, there will be no data to train or display!

- **Environment variables not loading:**  
  - Make sure `python-dotenv` is installed (`pip install python-dotenv`).
  - The `.env` file must be in your project root.

---

## Code Quality

<<<<<<< HEAD
- **Formatting:** Use [Black](https://black.readthedocs.io/en/stable/) for code formatting:  
  ```bash
  black .
 
=======
- Flask
- pandas, numpy, scikit-learn, xgboost
- pymongo, python-dotenv, joblib, Faker

Install all with:

```bash
pip install -r requirements.txt
```

---

## Deployment

- The app uses the `PORT` environment variable if set (for Render/Heroku compatibility).
- For production, secure your Flask app and never commit your `.env` or credentials.

---
>>>>>>> cffb5a0605ff4fc8b5f6eebc7741e4a3a77c7a14
