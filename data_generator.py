import pandas as pd
import random
from faker import Faker
from pymongo import MongoClient
import os

# Initialize Faker for realistic data
fake = Faker()

def generate_claim_data(n=5000):
    """Generate synthetic motor insurance claims data."""
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
            "total_claim_amount": random.randint(1000, 20000),
            "fraud_reported": random.choice(["Yes", "No"])
        }
        data.append(claim)

    return pd.DataFrame(data)

def store_data_in_mongodb(df):
    """Store generated data into MongoDB Atlas."""
    mongo_uri = os.getenv("MONGO_URI")  # Ensure environment variable is set
    client = MongoClient(mongo_uri)
    db = client["claims-fraud-db"]
    collection = db["motor_insurance_claims"]

    # Insert data into MongoDB
    collection.insert_many(df.to_dict(orient="records"))
    print("✅ Data successfully stored in MongoDB Atlas")

if __name__ == "__main__":
    # Generate dataset
    df = generate_claim_data()
    df.to_csv("motor_insurance_claims.csv", index=False)
    print("✅ Data generation complete: Saved to motor_insurance_claims.csv")

    # Store in MongoDB
    store_data_in_mongodb(df)