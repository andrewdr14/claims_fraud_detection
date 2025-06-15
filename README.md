# Claims Fraud Detection

A data-driven web application that detects and analyzes insurance claim fraud using machine learning models (Random Forest, XGBoost) and MongoDB for data storage. The project provides a Flask-based web interface to evaluate model performance and review fraud statistics.

## Features

- **Synthetic Data Generation**: Create realistic motor insurance claim datasets with Faker.
- **Data Storage**: Store and retrieve claim data from MongoDB Atlas.
- **Model Training**: Train RandomForest and XGBoost classifiers to detect fraud.
- **Web Dashboard**: Visualize classification metrics and summary statistics via Flask.
- **Downloadable Data**: Users can download the generated dataset from the web UI.

## Project Structure

```
├── app.py                # Flask web app for evaluation and reporting
├── model.py              # Model training, saving/loading, and database logic
├── data_generator.py     # Synthetic data generator and MongoDB uploader
├── requirements.txt      # Python dependencies
├── motor_insurance_claims.csv # Example generated dataset
├── templates/
│   └── results.html      # HTML template for results display
├── .env                  # Environment variables (not tracked by git)
```

## Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/andrewdr14/claims-fraud-detection.git
cd claims-fraud-detection
```

### 2. Set up Python environment

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 3. Configure Environment Variables

Create a `.env` file in the project root:

```
MONGO_URI=your_mongodb_connection_string
```

### 4. Generate Data and Store in MongoDB

```bash
python data_generator.py
```

This will:
- Create `motor_insurance_claims.csv`
- Upload data to MongoDB

### 5. Train Models

Model training happens automatically the first time you run the app or `model.py`. Models will be saved as `random_forest.pkl` and `xgboost.pkl`.

### 6. Run the Web App

```bash
python app.py
```

Visit `http://localhost:8080` in your browser to see the dashboard.

## Usage

- **Home page (`/`)**: View model evaluation (classification reports) and summary statistics.
- **Download data (`/download-data`)**: Download the current dataset as CSV.

## Dependencies

- Python 3.7+
- Flask
- pandas, numpy, scikit-learn, xgboost
- pymongo, python-dotenv, joblib, Faker

Install all dependencies with:

```bash
pip install -r requirements.txt
```

## Deployment

The app is Render/Heroku-ready: it uses the `PORT` environment variable if set.

## Notes & Recommendations

- Ensure your MongoDB Atlas instance is accessible from your host.
- Do not commit sensitive credentials or your `.env` file.
- For production, secure your Flask app (see Flask docs).


