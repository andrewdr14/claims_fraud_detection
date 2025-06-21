"""
data_generator.py

This script generates a synthetic motor insurance claims dataset and saves it as a CSV.
It does NOT interact with MongoDB. To regenerate data, simply run this script.

Usage:
    python claims_fraud/data_generator.py
"""

import os
import pandas as pd
import numpy as np

def generate_claim_data(num_samples=1000):
    """
    Generates a synthetic dataset for motor insurance claims.

    Args:
        num_samples (int): Number of samples to generate.

    Returns:
        pd.DataFrame: The generated dataset.
    """
    np.random.seed(42)
    data = {
        'age': np.random.randint(18, 80, num_samples),
        'annual_premium': np.random.randint(20000, 100000, num_samples),
        'policy_sales_channel': np.random.randint(1, 20, num_samples),
        'vintage': np.random.randint(10, 300, num_samples),
        'fraud_reported': np.random.choice([0, 1], num_samples, p=[0.95, 0.05])
    }
    df = pd.DataFrame(data)
    return df

if __name__ == "__main__":
    # Generate the synthetic data
    df = generate_claim_data()
    # Save to CSV in same directory as script
    output_path = os.path.join(os.path.dirname(__file__), "motor_insurance_claims.csv")
    df.to_csv(output_path, index=False)
    print(f"✅ Data generation complete: Saved to {output_path}")