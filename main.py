from fastapi import FastAPI, File, Form, UploadFile
from pydantic import BaseModel
import numpy as np
import pandas as pd
from io import StringIO
import requests

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
    data_in = [[data['sepal_length'], data['sepal_width'], data['petal_length'], data['petal_width']]]
    inference_request = {"dataframe_records": data_in}
    endpoint = "http://localhost:1234/invocations"
    response = requests.post(endpoint, json=inference_request)
    print(response)
    return {'prediction': response.text}

@app.post("/files/")
async def batch_prediction(file: bytes = File(...)):
    s = str(file, 'utf-8')
    data = StringIO(s)
    df = pd.read_csv(data)
    lst = df.values.tolist()
    inference_request = {"dataframe_records": lst}
    endpoint = "http://localhost:1234/invocations"
    response = requests.post(endpoint, json=inference_request)
    print(response)
    return response.text



 