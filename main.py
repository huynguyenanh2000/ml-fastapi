# Put the code for your API here.
from typing import Union, Literal
from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import pandas as pd
import joblib
from starter.ml.data import process_data
from starter.ml.model import compute_model_metrics, inference
import uvicorn
import os

if "DYNO" in os.environ and os.path.isdir(".dvc"):
    os.system("dvc config core.no_scm true")
    if os.system("dvc pull") != 0:
        exit("dvc pull failed")
    os.system("rm -r .dvc .apt/usr/lib/dvc")

# Load the encoder and label binarizer
encoder = joblib.load('./model/encoder.pkl')
lb = joblib.load('./model/lb.pkl')

# Load the model
model = joblib.load('./model/logistic-regression.sav')


app = FastAPI()


class DataInput(BaseModel):
    workclass: str
    
    education: str
    
    marital_status: str
    
    occupation: str
    
    relationship: str
    
    race: str

    sex: str

    native_country: str
    
    age: int
    fnlgt: int
    education_num: int
    capital_gain: int
    capital_loss: int
    hours_per_week: int

    class config:
        schema_extra = {
            "example": {
                "age": 36,
                "workclass": "Private",
                "fnlgt": 302146,
                "education": "HS-grad",
                "education_num": 9,
                "marital_status": "Divorced",
                "occupation": "Craft-repair",
                "relationship": "Husband",
                "race": "White",
                "sex": "Male",
                "capital_gain": 2000,
                "capital_loss": 0,
                "hours_per_week": 45,
                "native_country": "United-States"
            }
        }

@app.get("/")
async def root():
    return {
        "message": "Welcome to the Adult Income Prediction API"
    }

# ...
# /predict Endpoint
@app.post("/predict")
async def predict(data: DataInput):
    dat = {
        "age": [data.age],
        "workclass": [data.workclass],
        "fnlgt": [data.fnlgt],
        "education": [data.education],
        "education_num": [data.education_num],
        "marital_status": [data.marital_status],
        "occupation": [data.occupation],
        "relationship": [data.relationship],
        "race": [data.race],
        "sex": [data.sex],
        "capital_gain": [data.capital_gain],
        "capital_loss": [data.capital_loss],
        "hours_per_week": [data.hours_per_week],
        "native_country": [data.native_country],
    }

    df = pd.DataFrame(dat)

    # Ensure categorical fields are treated as strings in DataFrame
    categorical_fields = ["workclass", "education", "marital_status", "occupation", "relationship", "race", "sex", "native_country"]
    for field in categorical_fields:
        df[field] = df[field].astype(str)

    # Use the correct order of categorical fields for the LabelBinarizer
    X, _, _, _ = process_data(df, categorical_features=categorical_fields, encoder=encoder, lb=lb, training=False)

    prediction = inference(model, X)

    # Inverse transform the prediction using the LabelBinarizer
    pred = lb.inverse_transform(prediction)

    return {
        "prediction": pred[0]
    }


if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)






    
  
                    
    





