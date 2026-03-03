import json
from pathlib import Path
from typing import List
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import predict

class Car(BaseModel):
    brand: str
    model: str
    year: int
    engine_size: str
    hp: str
    fuel_type: str
    km: str
    status: str


dataPath = Path(__file__).resolve().parent.parent / 'data'

try:
    with open(dataPath / 'eval_metrics.json', "r", encoding="utf-8") as f:
        eval_metrics = json.load(f)
except FileNotFoundError:
    eval_metrics = None

try:
    with open(dataPath / 'cars_list.json', "r", encoding="utf-8") as f:
        cars_list = json.load(f)
except FileNotFoundError:
    cars_list = None
app = FastAPI()


@app.post('/model/{model_name}/predict')
def car_predict(model_name: str, car_data: List[Car]):
    for car in car_data:
        if (car.brand + "_" + car.model) not in cars_list:
            raise HTTPException(404, detail=f'{car.brand}_{car.model} was not found in available cars list')
    predicted_values = predict.predict(model_name, car_data)
    return [{**car_data[i].model_dump(), 'price': float(predicted_values[i])} for i in range(0, len(car_data))]

@app.get('/model/{model_name}/metrics')
def get_metrics(model_name: str):
    if eval_metrics is not None:
        if model_name in eval_metrics:
            return eval_metrics[model_name]
        else:
            raise HTTPException(404, detail='Model was not found')
    else:
        raise HTTPException(404, detail='Metrics was not found')

@app.get('/model/available_cars')
def get_available_cars():
    if cars_list is not None:
        return cars_list
    else:
        raise HTTPException(404, detail='Cars list was not found')
