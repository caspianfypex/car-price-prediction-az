# 🚗 Car Price Prediction (Azerbaijan/Turbo.AZ)

**Predict used car prices in Azerbaijan** using Machine Learning.  
This project trains and evaluates regression models (XGBoost & Random Forest) to estimate market prices based on car attributes.


## 🚀 Project Overview
This repository contains a machine learning pipeline for predicting car prices, specifically focusing on the Turbo.az, Azerbaijan car market. The models are trained using common vehicle features (year, mileage, brand, etc.) and evaluated using regression metrics.

## 🔨 Installation  
1. Clone the repository:
```
git clone https://github.com/caspianfypex/car-price-prediction-az.git
 ```
2. Environment Setup
```
python -m venv .venv
source .venv/bin/activate  # macOS/Linux
.venv\Scripts\activate    # Windows
``` 
3. Install dependencies:
```
pip install -r requirements.txt
```
4. Scraping (Optional):
```
python src/scraper.py
```
5. Training (Optional):
```
python src/train.py
```
6. Download RandomForest Model(Optional)
  
Pre-trained RandomForest model is not included due to huge size, should be [downloaded here](https://drive.google.com/file/d/1CgGX3KRq9WafmW9kFtxEKfjcWZFW42w9/view) and put in ```models/``` directory<br>

7. Running API:
```
python -m uvicorn src.app:app --reload
```

## 🛠️ API Usage
Details can be accessed through [http://localhost:8000/docs](http://localhost:8000/docs)

### Predictions:
Endpoint: POST /model/{model_name}/predict<br>
model_name: "xgb" or "rf"<br>
Request body: List of cars (JSON)<br>

**Example Request:**
```
[
  {
    "brand": "Mercedes",
    "model": "GLS 450 4MATIC",
    "year": 2023,
    "engine_size": "3.0 L",
    "hp": "381 a.g.",
    "fuel_type": "Benzin",
    "km": "65 055 km",
    "status": "Vuruğu yoxdur, rənglənməyib"
  }
]
```
**Example Response:**
```
[
  {
    "brand": "Mercedes",
    "model": "GLS 450 4MATIC",
    "year": 2023,
    "engine_size": "3.0 L",
    "hp": "381 a.g.",
    "fuel_type": "Benzin",
    "km": "65 055 km",
    "status": "Vuruğu yoxdur, rənglənməyib",
    "price": 130288.90
  }
]
```

## 📂 Project Structure
```
car-price-prediction-az/
├── data/                      # Raw/processed data
├── models/                    # Saved model files after training (XGBoost/RandomForest)
├── plots/                     # Visualization plots
├── src/                       # Source code
│   ├── app.py                 # API to send requests and make predictions
│   ├── predict.py             # Prediction
│   ├── preprocessing.py       # Feature engineering and preprocessing of data
│   ├── scraper.py             # Scraping local car market data from Turbo.az
│   └── train.py               # Training
├── .gitignore                 # Git ignore configuration
├── README.md                  # Project overview & documentation
└── requirements.txt           # Dependencies
```

## 📁 Dataset

The dataset includes used car listings with features such as:
| Feature | Description |
|---------|-------------|
| `brand` | Car manufacturer |
| `model` | Car model |
| `year` | Year of production |
| `kilometrage` | Driven distance (km) |
| `engine_size` | Engine size |
| `horse_power` | Horse power |
| `fuel_type` | Petrol/Diesel/Electric/Hybrid |
| `status` | Accident/Repaired/Painted |
| `price` | Target variable (AZN) |

#### Used Dataset Description<br>
For training, data scraped from Turbo.az was used. You may use your own properly formatted dataset or scrape data on your local device<br>
**Scrape Date**: February 2026<br>

Models appearing less than 30 times in the dataset were excluded during training. The metrics below show results after this step.

| Metric | Value |
|---------|-------------|
| Price Mean (AZN) | 30,445 |
| Price Std (AZN) | 39,777 |
| Total Data Size | 55,993 |
| Train Data Size (80%) | 44,794 |
| Test Data Size (20%) | 11,199 |

## 🔍 Data Cleaning and Feature Engineering

Before training, following steps were done:

- Handling missing values  
- Encoding categorical variables  
- Visualizing target distribution  
- Feature engineering: added KMperYear and Age
- Removal of models that appear less than 30 times

Example plots are stored in the `plots/` directory.

---

## ⚙️ Models

This project trains two regressors:

### 🧩 XGBoost Regressor

**Hyperparameters**
```python
  learning_rate: 0.05,
  max_depth: 6,
  n_estimators: 1000,
  tree_method='hist'
```

**Evaluation Metrics**
| Metric | Value |
|---------|-------------|
| `RMSE` | 5250.68 |
| `MAE` | 2525.49 |
| `MAPE` | 10.65% |
| `R²` | 0.9812 |

### 🧩 RandomForest Regressor

**Hyperparameters**
```python
  n_estimators=500,
  min_samples_split=5,
  min_samples_leaf=2,
  max_features='sqrt',
  bootstrap=True
```

**Evaluation Metrics**
| Metric | Value |
|---------|-------------|
| `RMSE` | 5513.22 |
| `MAE` | 2331.71 |
| `MAPE` | 10.14% |
| `R²` | 0.9793 |

Note: Model was not included in ```models/``` due to its huge size, can be trained, or accessed to pre-trained model through [download link](https://drive.google.com/file/d/1CgGX3KRq9WafmW9kFtxEKfjcWZFW42w9/view).

## 📝 Conclusion
Both XGBoost and Random Forest performed well on dataset with high accuracy and low prediction errors. XGBoost showed a lower RMSE, meaning it handled larger price deviations better, while Random Forest showed a lower MAE, indicating more consistent average prediction accuracy.

To improve model stability and reduce noise, only car models that appeared more than 30 times in the dataset were included in the training process. This filtering step helped the models learn more reliable patterns, and improved overall performance.

Predictions are currently limited to car brands and models listed in ```data/cars_list.json```, as the model was trained only on those categories.

## ⚠️ **Legal Disclaimer** 
This project was trained on data scraped from Turbo.az. Users can scrape data privately on their own device, but redistributing or sharing scraped data publicly may violate legal or copyright restrictions. Use at your own risk.
