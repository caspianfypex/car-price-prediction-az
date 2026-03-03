import json
from pathlib import Path
import pandas as pd
from pandas import DataFrame
import joblib
from sklearn.preprocessing import LabelEncoder

dataPath = Path(__file__).resolve().parent.parent / 'data'
try:
    model_brand_le = joblib.load(dataPath / 'label_encoder.pkl')
except FileNotFoundError:
    model_brand_le = LabelEncoder()

def prepare_data(data: DataFrame) -> DataFrame:
    data['Year'] = data['Year'].astype(dtype=int)
    data['Horse Power'] = data['Horse Power'].str.replace(' a.g.', '', regex=False)
    data['Horse Power'] = pd.to_numeric(data['Horse Power'], errors='coerce').astype('Int64')
    data['Kilometrage'] = data['Kilometrage'].str.replace(r' km| ', '', regex=True)
    data['Kilometrage'] = data['Kilometrage'].astype(dtype=int)
    data['Engine Size'] = data['Engine Size'].str.replace(' L', '', regex=False)
    data['Engine Size'] = data['Engine Size'].fillna("0")
    data['Engine Size'] = data['Engine Size'].apply(lambda x: float(x.replace(' sm3', ''))/1000 if " sm3" in x else x).astype(float)
    if 'Price' in data.columns:
        data = data.dropna()
        data['Price'] = data['Price'].str.replace(r' |₼|≈', '', regex=True)
        data['Price'] = data['Price'].astype(dtype=int)

        brand_counts = data['Brand'].value_counts()
        data = data[data['Brand'].isin(brand_counts[brand_counts >= 30].index)]
        model_counts = data['Model'].value_counts()
        data = data[data['Model'].isin(model_counts[model_counts >= 30].index)]
    luxury_brands = [
        'Mercedes', 'BMW', 'Audi', 'Porsche', 'Lexus', 'Land Rover',
        'Jaguar', 'Bentley', 'Rolls-Royce', 'Tesla', 'Maserati', 'Genesis',
        'Cadillac', 'Lincoln', 'Infiniti', 'ZEEKR', 'Li Auto', 'Hongqi'
    ]
    mid_level_brands = [
        'Toyota', 'Hyundai', 'Kia', 'Volkswagen', 'Ford', 'Nissan', 'Honda',
        'Chevrolet', 'Mitsubishi', 'Mazda', 'Skoda', 'Opel', 'Renault',
        'Peugeot', 'Jeep', 'Subaru', 'BYD', 'Changan', 'Haval', 'Geely', 'Chery'
    ]
    data['Luxury'] = data['Brand'].isin(luxury_brands).astype(int)
    data['Mid-Level'] = data['Brand'].isin(mid_level_brands).astype(int)
    data = pd.get_dummies(data, columns=['Fuel Type', 'Status'], drop_first=True, dtype=int)
    data['Brand_Model'] = data['Brand'] + '_' + data['Model']
    if 'Price' in data.columns:
        with open(dataPath / 'cars_list.json', 'w', encoding='utf-8') as f:
            json.dump(list(data['Brand_Model'].unique()), f, indent=4, ensure_ascii=False)
        data['Brand_Model'] = model_brand_le.fit_transform(data['Brand_Model'])
        joblib.dump(model_brand_le, dataPath / 'label_encoder.pkl')
    else:
        data['Brand_Model'] = model_brand_le.transform(data['Brand_Model'])
    data['Age'] = 2026 - data['Year']
    data['KMPerYear'] = data['Kilometrage'] / (data['Age'] + 1)
    data = data.drop(columns=['Brand','Model','Year'])
    if 'Price' not in data.columns:
        training_columns = ['Engine Size', 'Horse Power', 'Kilometrage', 'Price', 'Luxury', 'Mid-Level', 'Fuel Type_Dizel',
       'Fuel Type_Dizel-Hibrid', 'Fuel Type_Elektro', 'Fuel Type_Hibrid',
       'Fuel Type_Plug-in Hibrid', 'Fuel Type_Qaz', 'Status_Vuruğu var, rənglənməyib',
       'Status_Vuruğu yoxdur, rənglənib',
       'Status_Vuruğu yoxdur, rənglənməyib', 'Brand_Model', 'Age',
       'KMPerYear']
        for col in training_columns:
            if col not in data.columns:
                data[col] = 0
        data = data[training_columns]
        data = data.drop(columns=['Price'])
    return data