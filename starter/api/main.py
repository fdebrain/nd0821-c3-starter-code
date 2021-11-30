import os
from pathlib import Path

import joblib
import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from starter.ml.data import process_data
from starter.ml.model import inference
from starter.train_model import CAT_FEATURES

app = FastAPI()
ROOT_DIR = Path(os.path.dirname(os.path.abspath(__file__))).parent

# DVC requirement
if "DYNO" in os.environ and os.path.isdir(".dvc"):
    os.system("dvc config core.no_scm true")
    if os.system("dvc pull") != 0:
        exit("dvc pull failed")
    os.system("rm -r .dvc .apt/usr/lib/dvc")


class ToHyphen(BaseModel):
    class Config:
        def to_hyphen(string):
            return string.replace('_', '-')

        alias_generator = to_hyphen
        allow_population_by_field_name = True


class InputData(ToHyphen):
    age: int = Field(..., example=40)
    workclass: str = Field(..., example='Private')
    fnlgt: int = Field(..., example=160187)
    education: str = Field(..., example='Masters')
    education_num: int = Field(..., example=14)
    marital_status: str = Field(..., example='Married-civ-spouse')
    occupation: str = Field(..., example='Prof-specialty')
    relationship: str = Field(..., example='Not-in-family')
    race: str = Field(..., example='White')
    sex: str = Field(..., example='Female')
    capital_gain: int = Field(..., example=40000)
    capital_loss: int = Field(..., example=0)
    hours_per_week: int = Field(..., example=40)
    native_country: str = Field(..., example='Canada')


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
