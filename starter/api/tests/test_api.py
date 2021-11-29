import json

from api.main import app
from fastapi.testclient import TestClient

client = TestClient(app)


def test_get_root():
    r = client.get("/")
    assert r.status_code == 200
    assert r.json() == {"Hello": "World"}


def test_predict_zero():
    data = {'age': 20,
            'workclass': 'Private',
            'fnlgt': 117037,
            'education': '11th',
            'education-num': 7,
            'marital-status': 'Married-civ-spouse',
            'occupation': 'Transport-moving',
            'relationship': 'Husband',
            'race': 'White',
            'sex': 'Male',
            'capital-gain': 0,
            'capital-loss': 2042,
            'hours-per-week': 40,
            'native-country': 'United-States'}

    r = client.post("/predict", data=json.dumps(data))
    assert r.status_code == 200
    assert r.json()["prediction"] == '0'


def test_predict_one():
    data = {'age': 50,
            'workclass': 'Private',
            'fnlgt': 367260,
            'education': 'Bachelors',
            'education-num': 13,
            'marital-status': 'Never-married',
            'occupation': 'Tech-support',
            'relationship': 'Unmarried',
            'race': 'White',
            'sex': 'Male',
            'capital-gain': 14084,
            'capital-loss': 0,
            'hours-per-week': 45,
            'native-country': 'Canada'}

    r = client.post("/predict", data=json.dumps(data))
    assert r.status_code == 200
    assert r.json()["prediction"] == '1'
