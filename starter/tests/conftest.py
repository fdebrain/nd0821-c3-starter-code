import pandas as pd
import pytest
from starter.ml.data import process_data
from starter.ml.model import train_model


@pytest.fixture
def cat_features():
    return ["workclass",
            "education",
            "marital-status",
            "occupation",
            "relationship",
            "race",
            "sex",
            "native-country"]


@pytest.fixture
def data():
    return pd.read_csv('data/census_cleaned.csv').sample(100)


@pytest.fixture
def processed_data(data, cat_features):
    X, y, _, _ = process_data(data,
                              categorical_features=cat_features,
                              label="salary",
                              training=True)
    return X, y


@pytest.fixture
def model(processed_data):
    X, y = processed_data
    return train_model(X, y)
