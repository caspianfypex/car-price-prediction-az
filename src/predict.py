from fastapi import HTTPException
from pathlib import Path
import numpy as np
from pandas import DataFrame
import preprocessing
import joblib

modelPath = str(Path(__file__).resolve().parent.parent) + '\\models\\'

try:
    model_xgb = joblib.load(f'{modelPath}model_xgb.pkl')
except FileNotFoundError:
    model_xgb = None
try:
    model_rf = joblib.load(f'{modelPath}model_rf.pkl')
except FileNotFoundError:
    model_rf = None

models = {'xgb': model_xgb, 'rf': model_rf}

def predict(model_name, car_list_data):
    if model_name not in models:
        raise HTTPException(404, detail='Model not found')
    car = DataFrame([x.model_dump().values() for x in car_list_data], columns=['Marka', 'Model', 'Il', 'L', 'At gucu', 'Fuel', 'KM', 'Veziyyet'])
    processed_data = preprocessing.prepare_data(car)
    model = models.get(model_name)
    predicted_values = model.predict(processed_data.to_numpy())
    return np.expm1(predicted_values)