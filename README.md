# 🚗 Car Price Prediction (Azerbaijan)

**Predict used car prices in Azerbaijan** using Machine Learning.  
This project trains and evaluates regression models (XGBoost & Random Forest) to estimate market prices based on car attributes.


## 🚀 Project Overview
This repository contains a machine learning pipeline for predicting car prices, specifically focusing on the Turbo.az, Azerbaijan used car market. The models are trained using common vehicle features (year, mileage, brand, etc.) and evaluated using regression metrics.


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

> 📌 Scraped data of Turbo.az was used for training. You can scrape data privately on your own device, but sharing publicly is forbidden. 

## 🔍 Data Cleaning and Feature Engineering

Before training, followings are done:

- Handling missing values  
- Encoding categorical variables  
- Visualizing target distribution  
- Addition of features, such as KMperYear and Age
- Removal of models that appear less than 30 times from dataset

Example EDA plots are stored in the `plots/` directory.

---

## ⚙️ Models and Hyperparameters

This project trains two regressors:

### 🧩 XGBoost Regressor

```python
  learning_rate: 0.05,
  max_depth: 6,
  n_estimators: 1000,
  tree_method='hist'

### 🧩 RandomForest Regressor

```python
  n_estimators=500,
  min_samples_split=5,
  min_samples_leaf=2,
  max_features='sqrt',
  bootstrap=True,
