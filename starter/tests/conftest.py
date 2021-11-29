import pandas as pd
import pytest
from starter.ml.data import process_data
from starter.ml.model import train_model
from starter.train_model import CAT_FEATURES


@pytest.fixture
def data():
    return pd.read_csv('./starter/data/census_cleaned.csv', nrows=100)


@pytest.fixture
def processed_data(data):
    X, y, _, _ = process_data(data,
                              categorical_features=CAT_FEATURES,
                              label="salary",
                              training=True)
    return X, y


@pytest.fixture
def model(processed_data):
    X, y = processed_data
    return train_model(X, y)
