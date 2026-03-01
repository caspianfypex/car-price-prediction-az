from pathlib import Path
from typing import List
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import predict
import joblib

class Car(BaseModel):
    brand: str
    model: str
    year: int
    engine_size: str
    hp: str
    fuel_type: str
    km: str
    status: str

dataPath = str(Path(__file__).resolve().parent.parent) + '\\data\\'
cars_list = joblib.load(f'{dataPath}cars_list.pkl')
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
    if model_name == 'xgb':
        return {"RMSE": 5250.68,"MAE": 2525.49,"MAPE": 10.65,"R2": 0.9812}
    elif model_name == 'rf':
        return {"RMSE": 5513.22, "MAE": 2331.71, "MAPE": 10.14, "R2": 0.9793}
    else:
        raise HTTPException(404, detail='Model was not found')

@app.get('/model/available_cars')
def get_available_cars():
    return cars_list