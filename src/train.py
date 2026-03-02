from pathlib import Path
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from preprocessing import prepare_data
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

rootPath = Path(__file__).resolve().parent.parent
dataPath = rootPath / 'data'
modelPath = rootPath / 'models'
plotsPath = rootPath / 'plots'


data = prepare_data(pd.read_csv(dataPath / 'data.csv'))

price = data['Price']
features = data.drop(columns=['Price'])

#Price Distribution Visualization
plt.figure(figsize=(10, 6))
plt.hist(price, bins=30,color='blue',edgecolor='black',alpha=0.7)
plt.title('Price Distribution')
plt.xlabel('Price')
plt.ylabel('Frequency')
plt.tight_layout()
plt.savefig(plotsPath / 'price.png', dpi=150)
#

x_train,x_test,y_train,y_test = train_test_split(features, price, test_size=0.2, shuffle=True, random_state=42)
x_train = x_train.to_numpy()
x_test = x_test.to_numpy()
y_train = np.log1p(y_train.to_numpy())
y_test = np.log1p(y_test.to_numpy())

model_xgb = XGBRegressor(
    n_estimators=1000,
    learning_rate=0.05,
    max_depth=6,
    tree_method='hist',
    random_state=42
)
model_rf = RandomForestRegressor(
    n_estimators=500,
    min_samples_split=5,
    min_samples_leaf=2,
    max_features='sqrt',
    bootstrap=True,
    n_jobs=-1,
    random_state=42
)


model_xgb.fit(x_train, y_train)
model_rf.fit(x_train, y_train)


y_pred = np.expm1(model_xgb.predict(x_test))
y_test = np.expm1(y_test)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
r2 = r2_score(y_test,y_pred)

print("===== Model Metrics =====")
print('Model: XGBoost Regressor')
print(f"RMSE: {rmse:.2f}")
print(f"MAE: {mae:.2f}")
print(f"MAPE: {mape:.2f} %")
print(f"R²: {r2:.4f}")

#XGB Data Visualization
plt.clf()
plt.scatter(y_test, y_pred, alpha=0.5)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted Price")
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--')
plt.savefig(plotsPath / 'xgb_data.png', dpi=150)
#

y_pred = np.expm1(model_rf.predict(x_test))

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
r2 = r2_score(y_test,y_pred)

print(' ')
print("===== Model Metrics =====")
print('Model: RandomForest Regressor')
print(f"RMSE: {rmse:.2f}")
print(f"MAE: {mae:.2f}")
print(f"MAPE: {mape:.2f} %")
print(f"R²: {r2:.4f}")

#Random Forest Data Visualization
plt.clf()
plt.scatter(y_test, y_pred, alpha=0.5)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted Price")
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--')
plt.savefig(plotsPath / 'rf_data.png', dpi=150)


joblib.dump(model_xgb, modelPath / 'model_xgb.pkl')
joblib.dump(model_rf, modelPath / 'model_rf.pkl')

#XGB Feature Importance Visualization
features = np.array(features.columns)
importances = model_xgb.feature_importances_

indices = np.argsort(importances)[::-1]
sorted_features = features[indices]
sorted_importances = importances[indices]
plt.clf()

plt.barh(range(len(sorted_importances)), sorted_importances, color='teal')
plt.yticks(range(len(sorted_features)), sorted_features)

plt.title('Feature Importance Bar')
plt.xlabel("Feature Importance")
plt.gca().invert_yaxis()

plt.tight_layout()
plt.savefig(plotsPath / 'xgb_feature_importance.png', dpi=150)
#

#Random Forest Feature Importance Visualization
importances = model_rf.feature_importances_

indices = np.argsort(importances)[::-1]
sorted_features = features[indices]
sorted_importances = importances[indices]
plt.clf()

plt.barh(range(len(sorted_importances)), sorted_importances, color='teal')
plt.yticks(range(len(sorted_features)), sorted_features)

plt.title('Feature Importance Bar')
plt.xlabel("Feature Importance")
plt.gca().invert_yaxis()

plt.tight_layout()
plt.savefig(plotsPath / 'rf_feature_importance.png', dpi=150)
#
