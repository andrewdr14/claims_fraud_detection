import pandas as pd
import random
from faker import Faker
from pymongo import MongoClient
import os
from dotenv import load_dotenv
from typing import Any

# Load environment variables from .env file
load_dotenv()

# Initialize Faker for realistic data
fake = Faker()

def is_fraudulent(
    claim: dict,
    base_fraud_prob=0.025,
    extra_prob_if_1=0.09,
    false_negative_prob=0.04
) -> str:
    """
    Assign fraud label with a rule-based system and some randomness.

    - If 2+ rules match: fraud unless marked as false negative by chance.
    - If 1 rule matches: fraud with a small probability (simulate false positives).
    - Otherwise: assign fraud randomly (low base rate).
    """
    suspicious = 0
    if claim['total_claim_amount'] > 15000 and claim['policy_deductible'] == 500:
        suspicious += 1
    if claim['incident_hour_of_the_day'] < 5 or claim['incident_hour_of_the_day'] > 22:
        suspicious += 1
    if claim['collision_type'] == "Hit & Run":
        suspicious += 1
    if claim['insured_age'] < 22 and claim['total_claim_amount'] > 12000:
        suspicious += 1
    if claim['number_of_vehicles'] >= 3 and claim['policy_annual_premium'] > 2500:
        suspicious += 1

    # Main rule: fraud if 2+ triggers, with a small chance (false_negative_prob) to escape detection
    if suspicious >= 2:
        return "No" if random.random() < false_negative_prob else "Yes"
    # 1 trigger: some chance of being fraud (false positive)
    elif suspicious == 1:
        return "Yes" if random.random() < extra_prob_if_1 else "No"
    # 0 triggers: very small chance of being fraud (random noise)
    else:
        return "Yes" if random.random() < base_fraud_prob else "No"

def generate_claim_data(n: int = 50000) -> pd.DataFrame:
    """
    Generate synthetic motor insurance claims data.

    Args:
        n (int, optional): Number of samples to generate. Defaults to 10000.

    Returns:
        pd.DataFrame: DataFrame containing synthetic claim records.
    """
    data = []
    collision_types = ["Rear-End", "Side Impact", "Front Collision", "Rollover", "Hit & Run"]

    for _ in range(n):
        claim = {
            "policy_number": f"POL{fake.random_int(min=10000, max=99999)}",
            "policy_deductible": random.choice([500, 1000, 2000]),
            "policy_annual_premium": random.randint(500, 3000),
            "umbrella_limit": random.choice([50000, 100000, 150000]),
            "insured_age": random.randint(18, 85),
            "incident_hour_of_the_day": random.randint(0, 23),
            "collision_type": random.choice(collision_types),
            "number_of_vehicles": random.randint(1, 5),
            "total_claim_amount": random.randint(1000, 20000)
        }
        claim["fraud_reported"] = is_fraudulent(claim)
        data.append(claim)

    return pd.DataFrame(data)

def store_data_in_mongodb(df: pd.DataFrame) -> None:
    """
    Store generated data into MongoDB Atlas.

    Args:
        df (pd.DataFrame): DataFrame containing the claim data to store.

    Raises:
        ValueError: If the MONGO_URI environment variable is not set.
    """
    mongo_uri = os.getenv("MONGO_URI")  # Ensure environment variable is set
    print("MONGO_URI is:", mongo_uri)   # Debug: Show the URI being used

    if not mongo_uri:
        raise ValueError("MONGO_URI environment variable not set. Please check your .env file.")

    try:
        client = MongoClient(mongo_uri)
        db = client["claims-fraud-db"]
        collection = db["motor_insurance_claims"]

        # Clear the collection before inserting new data
        collection.delete_many({})
        
        # Insert data into MongoDB
        collection.insert_many(df.to_dict(orient="records"))
        print("✅ Data successfully stored in MongoDB Atlas")
    except Exception as e:
        print("❌ Failed to store data in MongoDB Atlas:", e)

if __name__ == "__main__":
    # Generate dataset
    df = generate_claim_data()
    df.to_csv("motor_insurance_claims.csv", index=False)
    print("✅ Data generation complete: Saved to motor_insurance_claims.csv")

    # Store in MongoDB
    store_data_in_mongodb(df)