import json

import requests

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


response = requests.post('http://127.0.0.1:8000/predict', data=json.dumps(data))

print(response.status_code)
print(response.json())
