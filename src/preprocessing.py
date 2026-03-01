from pathlib import Path
import pandas as pd
from pandas import DataFrame
import joblib
from sklearn.preprocessing import LabelEncoder

dataPath = str(Path(__file__).resolve().parent.parent) + '\\data\\'
try:
    model_brand_le = joblib.load(f'{dataPath}label_encoder.pkl')
except FileNotFoundError:
    model_brand_le = LabelEncoder()

def prepare_data(data: DataFrame) -> DataFrame:
    data['Il'] = data['Il'].astype(dtype=int)
    data['At gucu'] = data['At gucu'].str.replace(' a.g.', '', regex=False)
    data['At gucu'] = pd.to_numeric(data['At gucu'], errors='coerce').astype('Int64')
    data['KM'] = data['KM'].str.replace(r' km| ', '', regex=True)
    data['KM'] = data['KM'].astype(dtype=int)
    data['L'] = data['L'].str.replace(' L', '', regex=False)
    data['L'] = data['L'].fillna("0")
    data['L'] = data['L'].apply(lambda x: float(x.replace(' sm3', ''))/1000 if " sm3" in x else x).astype(float)
    if 'Qiymet' in data.columns:
        data = data.dropna()
        data['Qiymet'] = data['Qiymet'].str.replace(r' |₼|≈', '', regex=True)
        data['Qiymet'] = data['Qiymet'].astype(dtype=int)

        brand_counts = data['Marka'].value_counts()
        data = data[data['Marka'].isin(brand_counts[brand_counts >= 30].index)]
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
    data['Luxury'] = data['Marka'].isin(luxury_brands).astype(int)
    data['Mid-Level'] = data['Marka'].isin(mid_level_brands).astype(int)
    data = pd.get_dummies(data, columns=['Fuel', 'Veziyyet'], drop_first=True, dtype=int)
    data['Marka_Model'] = data['Marka'] + '_' + data['Model']
    if 'Qiymet' in data.columns:
        joblib.dump(list(data['Marka_Model'].unique()), f'{dataPath}cars_list.pkl')
        data['Marka_Model'] = model_brand_le.fit_transform(data['Marka_Model'])
        joblib.dump(model_brand_le, f'{dataPath}label_encoder.pkl')
    else:
        data['Marka_Model'] = model_brand_le.transform(data['Marka_Model'])
    data['Age'] = 2026 - data['Il']
    data['KMPerYear'] = data['KM'] / (data['Age'] + 1)
    data = data.drop(columns=['Marka','Model','Il'])
    if 'Qiymet' not in data.columns:
        training_columns = ['L', 'At gucu', 'KM', 'Qiymet', 'Luxury', 'Mid-Level', 'Fuel_Dizel',
       'Fuel_Dizel-Hibrid', 'Fuel_Elektro', 'Fuel_Hibrid',
       'Fuel_Plug-in Hibrid', 'Fuel_Qaz', 'Veziyyet_Vuruğu var, rənglənməyib',
       'Veziyyet_Vuruğu yoxdur, rənglənib',
       'Veziyyet_Vuruğu yoxdur, rənglənməyib', 'Marka_Model', 'Age',
       'KMPerYear']
        for col in training_columns:
            if col not in data.columns:
                data[col] = 0
        data = data[training_columns]
        data = data.drop(columns=['Qiymet'])
    return data