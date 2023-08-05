from fastapi import FastAPI, File, Form, UploadFile
from pydantic import BaseModel
import numpy as np
import pandas as pd
from io import StringIO
import requests
import pickle

app = FastAPI()

class IrisSpecies(BaseModel):
    sepal_length: float 
    sepal_width: float 
    petal_length: float 
    petal_width: float

@app.get('/')
async def root():
    return {'message': 'Hello Midhilesh'}

@app.post('/predict')
async def individual_prediction(iris: IrisSpecies):
    data = iris.dict()
    loaded_model = pickle.load(open('LRClassifier.pkl', 'rb'))
    data_in = [[data['sepal_length'], data['sepal_width'], data['petal_length'], data['petal_width']]]
    prediction = loaded_model.predict(data_in)
    probability = loaded_model.predict_proba(data_in).max()

    return {'prediction' : prediction[0], 'probability' : probability}

@app.post("/files/")
async def batch_prediction(file: bytes = File(...)):
    s = str(file, 'utf-8')
    data = StringIO(s)
    df = pd.read_csv(data)
    loaded_model = pickle.load(open('LRClassifier.pkl', 'rb'))
    prediction = loaded_model.predict(df)
    probability = loaded_model.predict_proba(df).max()

    result = pd.DataFrame()
    result['pred_class'] = prediction
    result['setosa_prob'] = probability[:,0]
    result['versicolor_prob'] = probability[:,1]
    result['virginica_prob'] = probability[:,2]

    return result


 