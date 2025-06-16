import pytest
import pandas as pd
from claims_fraud import model

def test_fetch_data_returns_dataframe():
    """
    Test that fetch_data returns a pandas DataFrame
    and that it contains expected columns.
    """
    df = model.fetch_data()
    assert isinstance(df, pd.DataFrame)
    assert "fraud_reported" in df.columns

def test_initialize_models_trains_and_saves():
    """
    Test that initialize_models runs without error
    and sets the trained_features list.
    """
    model.initialize_models()
    assert isinstance(model.trained_features, list)
    assert len(model.trained_features) > 0