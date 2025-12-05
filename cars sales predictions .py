import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error



file_path = r'C:\Users\DELL\Downloads\car_sales_simulated.csv'
df = pd.read_csv(file_path)



df['gas_sales'] = df['gas_sales'].fillna(df['gas_sales'].median())
df['ev_sales'] = df['ev_sales'].fillna(df['ev_sales'].median())
df['hybrid_sales'] = df['hybrid_sales'].fillna(df['hybrid_sales'].median())
df['oil_price'] = df['oil_price'].fillna(df['oil_price'].median())


df['total_alternatives_sales'] = df['ev_sales'] + df['hybrid_sales']
df['oil_x_ev'] = df['oil_price'] * df['ev_sales']
df['log_oil_price'] = np.log1p(df['oil_price'])



for lag in range(1, 5):
 df[f'gas_sales_lag{lag}'] = df['gas_sales'].shift(lag).fillna(df['gas_sales'].median())
 df[f'oil_price_lag{lag}'] = df['oil_price'].shift(lag).fillna(df['oil_price'].median())


df['gas_sales_roll3'] = df['gas_sales'].rolling(3).mean().fillna(df['gas_sales'].median())
df['gas_sales_roll5'] = df['gas_sales'].rolling(5).mean().fillna(df['gas_sales'].median())
df['oil_price_roll3'] = df['oil_price'].rolling(3).mean().fillna(df['oil_price'].median())
df['oil_price_roll5'] = df['oil_price'].rolling(5).mean().fillna(df['oil_price'].median())



features = [
'oil_x_ev', 'total_alternatives_sales', 'log_oil_price',
'gas_sales_lag1', 'gas_sales_lag2', 'gas_sales_lag3', 'gas_sales_lag4',
'oil_price_lag1', 'oil_price_lag2', 'oil_price_lag3', 'oil_price_lag4',
'gas_sales_roll3', 'gas_sales_roll5', 'oil_price_roll3', 'oil_price_roll5'
]

X = df[features]
Y = np.log1p(df['gas_sales'])



X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, shuffle=False)



LRM = LinearRegression()
LRM.fit(X_train, Y_train)
Y_pred_LR = np.expm1(LRM.predict(X_test))
Y_test_actual = np.expm1(Y_test)

print("Linear Regression:")
print("R2:", r2_score(Y_test_actual, Y_pred_LR))
print("MAE:", mean_absolute_error(Y_test_actual, Y_pred_LR))
print("RMSE:", np.sqrt(mean_squared_error(Y_test_actual, Y_pred_LR)))
print()


rf = RandomForestRegressor(n_estimators=500, max_depth=15, random_state=42, n_jobs=-1)
rf.fit(X_train, Y_train)
Y_pred_RF = np.expm1(rf.predict(X_test))

print("Random Forest:")
print("R2:", r2_score(Y_test_actual, Y_pred_RF))
print("MAE:", mean_absolute_error(Y_test_actual, Y_pred_RF))
print("RMSE:", np.sqrt(mean_squared_error(Y_test_actual, Y_pred_RF)))
print()



XGB_model = XGBRegressor(
n_estimators=600, learning_rate=0.02, max_depth=4,
subsample=0.8, colsample_bytree=0.8, random_state=42
)
XGB_model.fit(X_train, Y_train)
Y_pred_XGB = np.expm1(XGB_model.predict(X_test))

print("XGBoost:")
print("R2:", r2_score(Y_test_actual, Y_pred_XGB))
print("MAE:", mean_absolute_error(Y_test_actual, Y_pred_XGB))
print("RMSE:", np.sqrt(mean_squared_error(Y_test_actual, Y_pred_XGB)))
print()


importances = rf.feature_importances_
print("Feature Importances:")
for f, imp in sorted(zip(features, importances), key=lambda x: x[1], reverse=True):
 print(f"{f}: {imp:.4f}")

