import pandas as pd
from claims_fraud import data_generator

def test_generate_claim_data_shape():
    """
    Test that generate_claim_data returns a DataFrame of the requested size
    and contains all required columns.
    """
    df = data_generator.generate_claim_data(100)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 100
    required_columns = [
        "policy_number", "policy_deductible", "policy_annual_premium",
        "umbrella_limit", "insured_age", "incident_hour_of_the_day",
        "collision_type", "number_of_vehicles", "total_claim_amount", "fraud_reported"
    ]
    for col in required_columns:
        assert col in df.columns

def test_generate_claim_data_values():
    """
    Test that generated data contains only valid values for key fields.
    """
    df = data_generator.generate_claim_data(10)
    assert set(df["fraud_reported"].unique()).issubset({"Yes", "No"})
    assert df["policy_deductible"].min() >= 500
    assert df["policy_deductible"].max() <= 2000