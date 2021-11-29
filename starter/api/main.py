import os
from pathlib import Path

import joblib
import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from starter.ml.data import process_data
from starter.ml.model import inference
from starter.train_model import CAT_FEATURES

app = FastAPI()
ROOT_DIR = Path(os.path.dirname(os.path.abspath(__file__))).parent


class ToHyphen(BaseModel):
    class Config:
        def to_hyphen(string):
            return string.replace('_', '-')

        alias_generator = to_hyphen
        allow_population_by_field_name = True


class InputData(ToHyphen):
    age: int
    workclass: str
    fnlgt: int
    education: str
    education_num: int
    marital_status: str
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: int
    capital_loss: int
    hours_per_week: int
    native_country: str


model_dir = f'{ROOT_DIR}/model'
encoder = joblib.load(f'{model_dir}/encoder.pkl')
model = joblib.load(f'{model_dir}/trained_model.pkl')


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.post("/predict")
def get_prediction(input: InputData):
    input_df = pd.DataFrame.from_dict([input.dict(by_alias=True)])
    x, _, _, _ = process_data(input_df, categorical_features=CAT_FEATURES, training=False, encoder=encoder)
    pred = inference(model, x)

    if pred is None:
        raise HTTPException(status_code=400, detail="Model not found.")
    return {'prediction': str(pred[0])}


if __name__ == "__main__":
    uvicorn.run(app)
