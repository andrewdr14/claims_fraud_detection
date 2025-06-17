# Claims Fraud Detection Learning Tool

This project is designed as an educational resource for understanding and experimenting with machine learning techniques for insurance claims fraud detection. The codebase is modular, with ample opportunities for learning, extension, and hands-on exploration.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Key Technologies Used](#key-technologies-used)
- [Theory: RandomForest & XGBoost](#theory-randomforest--xgboost)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Environment Variables](#environment-variables)
- [Running the Application](#running-the-application)
- [Running Tests](#running-tests)
- [Database: MongoDB](#database-mongodb)
- [Contributing](#contributing)
- [License](#license)

---

## Project Overview

This repository demonstrates a full workflow for detecting fraudulent insurance claims using machine learning, including:

- **Data generation and preprocessing** (with `data_generator.py`)
- **Model training and evaluation** (with `model.py`)
- **Web API for fraud prediction** (with `app.py`, using Flask)
- **Storing and retrieving data via MongoDB**
- **Unit tests for data generation logic**

It is intended as a **learning tool**, so code and documentation are designed for clarity and exploration.

---

## Key Technologies Used

- **Python 3.10+** (3.10 or 3.11 strongly recommended)
- **Flask**: Lightweight web framework for building the API (`app.py`)
- **MongoDB**: NoSQL database for storing claims data and predictions
- **pymongo**: Python driver for MongoDB
- **scikit-learn**: Machine learning models and utilities
- **xgboost**: Advanced gradient boosting model
- **pytest**: For running unit tests

---

## Theory: RandomForest & XGBoost

### RandomForest

RandomForest is an ensemble machine learning technique that builds multiple decision trees and combines their predictions to improve accuracy and reduce overfitting. Each tree is trained with a random subset of the data and features.

- **Pros**: Handles non-linear data, reduces overfitting, interpretable feature importance
- **Cons**: Slower and less interpretable than a single tree

### XGBoost

XGBoost (Extreme Gradient Boosting) is an efficient implementation of gradient-boosted decision trees. It builds trees sequentially, with each new tree focusing on the errors of the previous trees.

- **Pros**: High predictive power, built-in regularization, handles missing data
- **Cons**: More complex, harder to tune

---

## Project Structure

```
claims_fraud_detection/
│
├── claims_fraud/                # Main package with core logic
│   ├── __init__.py
│   ├── data_generator.py        # Scripts for generating synthetic claims data
│   ├── model.py                 # Model training, saving, and loading logic
│   └── ...                      # (other modules)
├── app.py                       # Flask web API for predictions
├── requirements.txt             # Python dependencies
├── tests/                       # Unit tests (currently for data_generator.py)
│   ├── __init__.py
│   └── test_data_generator.py
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
    cd claims_fraud_detection
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
   - Start the MongoDB service (usually `mongod` on the command line)

---

## Environment Variables

The application expects a MongoDB connection string to be set in a `.env` file in the project root directory.

1. **Create a `.env` file in the root directory:**
    ```
    MONGODB_URI=mongodb://your-username:your-password@host:port/dbname
    ```
   Replace the value with your actual MongoDB connection details.

2. **(Optional) See `.env.example` for formatting guidance.**

---

## Running the Application

### 1. Generate Synthetic Data

Run the data generator to populate your MongoDB with synthetic claims data.
```bash
python -m claims_fraud.data_generator
```

### 2. Train Models

Train and save the machine learning models:
```bash
python -m claims_fraud.model
```
This will create and persist models (e.g., as `.pkl` files) for later use by the API.

### 3. Start the Flask API

Launch the web API to serve predictions:
```bash
python -m claims_fraud.app
```
The API will be available at [http://localhost:5000](http://localhost:8080) by default.

### 4. Make Predictions

You can send POST requests to endpoints such as `/predict` with claim data, and the API will return a fraud prediction.  
(See `app.py` for endpoint details and example payloads.)

---

## Running Tests

Unit tests are provided for the data generation logic (and can be extended for other modules).

To run all tests:
```bash
pytest
```
This will execute the tests in the `tests/` directory, currently focused on `data_generator.py`.

---

## Database: MongoDB

- Data (synthetic claims, predictions, etc.) is stored in MongoDB.
- The application expects a MongoDB connection string to be set in `.env` as `MONGODB_URI`. By default, a local MongoDB instance on `localhost:27017` is expected.
- You can configure MongoDB connection details in your `.env` file.

---

## Contributing

This repository is designed for **learning and experimentation**.  
Suggestions, improvements, and contributions are highly welcome!

- Fork the repo and submit a pull request.
- Open issues for bugs, questions, or feature requests.
- Ideas for new data features, model types, or explanatory materials are especially encouraged.
