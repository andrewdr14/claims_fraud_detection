# Claims Fraud Detection Learning Tool

This project is an educational resource for understanding and experimenting with machine learning techniques for insurance claims fraud detection. The codebase is modular, with example opportunities for learning, extension, and hands-on exploration.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Key Technologies Used](#key-technologies-used)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Environment Variables](#environment-variables)
- [Workflow: How to Use This Repo](#workflow-how-to-use-this-repo)
- [Database: MongoDB](#database-mongodb)
- [Contributing](#contributing)

---

## Project Overview

This repository demonstrates a complete workflow for detecting fraudulent insurance claims using machine learning, including:

- **Exploratory Data Analysis and Preprocessing** (with Jupyter notebook)
- **Model training, feature selection, evaluation, and reporting** (with `model.py`)
- **Interactive dashboard and visualizations** (with Streamlit, `app.py`)
- **Storing and retrieving data via MongoDB**

It is intended as a **learning tool**, so code and documentation are designed for clarity and exploration.

---

## Key Technologies Used

- **Python 3.10+** (3.10 or 3.11 strongly recommended)
- **Jupyter Notebook**: EDA and data preparation
- **Streamlit**: Interactive dashboard for results visualization
- **MongoDB**: NoSQL database for storing claims data and predictions
- **pymongo**: Python driver for MongoDB
- **scikit-learn**: Machine learning models and utilities
- **xgboost**: Advanced gradient boosting model
- **matplotlib**: For result visualization
- **python-dotenv**: For `.env` config management

---

## Project Structure

```
claims_fraud_detection/
│
├── claims_fraud/                  
│   ├── __init__.py
│   ├── model.py                   # Model training, feature selection, and reporting logic
│   ├── app.py                     # Streamlit web dashboard for visualizations
│   ├── eda_and_cleaning.ipynb     # Jupyter notebook for EDA and data cleaning
│   └── cleaned_insurance_claims.csv # Cleaned data produced by Jupyter notebook
|   └── insurance_claims.csv # Raw data used for the project
│
├── requirements.txt               # Python dependencies
├── README.md
└── ...
```

---

## Installation

> **Python 3.10 or 3.11 is strongly recommended.**  
> **MongoDB must also be installed and running locally or accessible remotely.**

1. **Clone the repository:**
    ```bash
    git clone https://github.com/andrewdr14/claims_fraud_detection.git
    cd claims_fraud_detection/claims_fraud
    ```

2. **(Recommended) Create and activate a virtual environment:**
    ```bash
    python -m venv venv
    # On Windows:
    venv\Scripts\activate
    # On macOS/Linux:
    source venv/bin/activate
    ```

3. **Install dependencies:**
    ```bash
    pip install --upgrade pip
    pip install -r requirements.txt
    ```

4. **Install and start MongoDB:**
    - [Download & Install MongoDB](https://docs.mongodb.com/manual/installation/)
    - Start the MongoDB service (usually `mongod` on the command line).

---

## Environment Variables

The application expects a MongoDB connection string to be set in a `.env` file in the `claims_fraud` directory (where `model.py` and `app.py` reside).

**Example `.env` file:**
```
MONGO_URI=mongodb://localhost:27017
```
Replace the value with your actual MongoDB connection details.

---

## Workflow: How to Use This Repo

#### 1. **Clone the repository**
```bash
git clone https://github.com/andrewdr14/claims_fraud_detection.git
cd claims_fraud_detection/claims_fraud
```

#### 2. **Set up your environment**
- *(See [Installation](#installation) above)*

#### 3. **Open and run the Jupyter Notebook for EDA and Data Cleaning**
- Start Jupyter in your terminal:
    ```bash
    jupyter notebook
    ```
- Open `eda_and_cleaning.ipynb`
- **Run all cells** to:
    - Explore and clean the data
    - Encode categorical variables
    - Save `cleaned_insurance_claims.csv` and `categorical_mappings.pkl` in your local directory

#### 4. **Set up your MongoDB environment file**
- Create a `.env` file in the `claims_fraud` directory and add your MongoDB connection string as shown above.

#### 5. **Run the modeling pipeline**
```bash
python model.py
```
- This will:
    - Perform feature selection and modeling
    - Save all output files (plots, reports, etc.) in your local directory for dashboard use

#### 6. **Launch the Streamlit dashboard**
```bash
streamlit run app.py
```
- The browser will open the dashboard with all results and visualizations.

---

## Database: MongoDB

- Synthetic or cleaned claim data and model predictions are stored in MongoDB.
- The application expects a MongoDB connection string as `MONGO_URI` in your `.env`.
- By default, a local MongoDB instance on `localhost:27017` is expected, but you can adjust this for remote/cloud MongoDB.

---

## Contributing

This repository is designed for **learning and experimentation**.  
Suggestions, improvements, and contributions are highly welcome!

- Fork the repo and submit a pull request.
- Open issues for bugs, questions, or feature requests.
- Ideas for new data features, model types, or explanatory materials are especially encouraged.
