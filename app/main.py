from fastapi import FastAPI
import joblib
import numpy as np
import pandas as pd
from pydantic import BaseModel
from typing import Optional

model = joblib.load("app/model/model_compressed.pkl")
num_pipeline = joblib.load("app/pipelines/num_pipeline.pkl")
cat_pipeline = joblib.load("app/pipelines/cat_pipeline.pkl")

app = FastAPI()

class X(BaseModel):
    year: float
    odometer: float
    cylinders: str
    manufacturer: str
    model: Optional[str] = np.nan
    condition: str
    fuel: str
    title_status: Optional[str] = np.nan
    transmission: str
    drive: str
    type: str

@app.get("/")
def read_root():
    return {"message": "Hello! This is a microservice for a model that predicts prices of used cars."}

@app.post("/predict")
def create_predictions(X: X):
    X_dict = X.dict()
    X_df = pd.DataFrame([X_dict])
    
    X_num = X_df[['year', 'odometer']] 
    X_cat = X_df[['manufacturer', 'model', 'condition', 'cylinders', 'fuel', 
                'transmission', 'title_status', 'drive', 'type']]

    X_num = num_pipeline.transform(X_num)
    X_cat = cat_pipeline.transform(X_cat)

    ct = cat_pipeline.named_steps['encoder']
    ohe = ct.named_transformers_['one_hot_encoder']
    ohe_feature_names = ohe.get_feature_names_out()

    cols = ['year', 'odometer','manufacturer', 'model','condition', 'cylinders']+list(ohe_feature_names)
          
    X = pd.DataFrame(np.concatenate((X_num, X_cat), axis=1), columns=cols)
    X = X[["year", "odometer", "cylinders", "manufacturer", "model", "condition", "x0_diesel", "x0_gas", "x1_automatic", "x1_manual", "x3_fwd", "x3_4wd", "x3_rwd", "x4_pickup", "x4_sedan", "x4_SUV"]]

    y = model.predict(X)

    return {"prediction": y[0]}