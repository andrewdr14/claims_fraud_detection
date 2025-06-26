# Claims Fraud Detection Learning Tool

This project is an educational resource for understanding and experimenting with machine learning techniques for insurance claims fraud detection. The codebase is modular, with example opportunities for learning, extension, and hands-on exploration.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Key Technologies Used](#key-technologies-used)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Environment Variables](#environment-variables)
- [Running the Application](#running-the-application)
- [Running Tests](#running-tests)
- [Database: MongoDB](#database-mongodb)
- [Contributing](#contributing)
- [Limitations](#limitations)
- [License](#license)

---

## Project Overview

This repository demonstrates a complete workflow for detecting fraudulent insurance claims using machine learning, including:

- **Data generation and preprocessing** (with `data_generator.py`)
- **Model training and evaluation** (with `model.py`)
- **Web API for fraud prediction** (with `app.py`, using Flask)
- **Storing and retrieving data via MongoDB**
- **Unit tests for data generation**
- **Interactive dashboard and visualizations**

It is intended as a **learning tool**, so code and documentation are designed for clarity and exploration.

---

## Key Technologies Used

- **Python 3.10+** (3.10 or 3.11 strongly recommended)
- **Flask**: Lightweight web framework for building the API (`app.py`)
- **MongoDB**: NoSQL database for storing claims data and predictions
- **pymongo**: Python driver for MongoDB
- **scikit-learn**: Machine learning models and utilities
- **xgboost**: Advanced gradient boosting model
- **pytest**: For running unit tests (currently focused on `data_generator.py`)
- **matplotlib**: For result visualization
- **Faker**: For generating realistic synthetic claim data

---

## Project Structure

```
claims_fraud_detection/
│
├── claims_fraud/                  
│   ├── __init__.py
│   ├── data_generator.py          # Scripts for generating synthetic claims data
│   ├── model.py                   # Model training, saving, and loading logic
│   ├── app.py                     # Flask web API for predictions
│   └── ...                        # (other modules)
│
├── requirements.txt               # Python dependencies
├── tests/                         # Unit tests (currently for data_generator.py)
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
    - Start the MongoDB service (usually `mongod` on the command line).

---

## Environment Variables

The application expects a MongoDB connection string to be set in a `.env` file in the project root directory.

**Example `.env` file:**
```
MONGO_URI=mongodb://localhost:27017
```
Replace the value with your actual MongoDB connection details.

---

## Running the Application

### 1. Generate Synthetic Data

Run the data generator to populate your MongoDB with synthetic claims data:

```bash
python -m claims_fraud.data_generator
```

### 2. Train Models

Train and save the machine learning models:

```bash
python -m claims_fraud.model
```

### 3. Start the Flask API

Launch the web API to serve predictions and visualizations:

```bash
python -m claims_fraud.app
```
The API will be available at [http://localhost:8080](http://localhost:8080) by default.

---

## Running Tests

Unit tests are provided for the data generation logic (and can be extended for other modules).

To run all tests:
```bash
pytest
```

---

## Database: MongoDB

- Synthetic claim data and predictions are stored in MongoDB.
- The application expects a MongoDB connection string as `MONGO_URI` in your `.env`.
- By default, a local MongoDB instance on `localhost:27017` is expected, but you can adjust this for remote or cloud MongoDB setups.

---

## Contributing

This repository is designed for **learning and experimentation**.  
Suggestions, improvements, and contributions are highly welcome!

- Fork the repo and submit a pull request.
- Open issues for bugs, questions, or feature requests.
- Ideas for new data features, model types, or explanatory materials are especially encouraged.

---

## Limitations

**This project is a learning tool with important limitations:**

- **Synthetic Data and Labels:**  
  All data is artificially generated using a mixture of business rules and randomness. Most fraudulent claims are labeled based on a small set of simple, transparent rules; a minority are labeled as fraud randomly (to simulate error/noise), and some true frauds are missed at random (simulating false negatives).

- **Feature–Label Leakage:**  
  The same features used to assign fraud labels are also used for model training. In real-world scenarios, the true fraud labels are not a direct function of input features, and "label leakage" is much less likely.

- **Clean Data:**  
  The synthetic data is well-formed, with no missing values, typos, or real-world messiness.

- **Class Balance:**  
  The proportion of fraudulent cases can be set arbitrarily; real insurance fraud is fairly rare.

- **No Adversarial Examples:**  
  In reality, fraudsters adapt and try to evade detection. Here, there is no such adversarial activity.

- **Overoptimistic Model Performance:**  
  Because of the above, models achieve unrealistically high metrics (precision, recall, ROC-AUC, etc.).  
  Real claims fraud detection is much more difficult, with far more subtle signals and noise.

---

**Suggestions & improvements are welcome!**
