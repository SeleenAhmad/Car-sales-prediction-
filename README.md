# Car-sales-prediction-
a study and cleaning of the data of hybrid and electric cars and oil prices and how it does affect the prices of gas dependent cars  
Car Sales Prediction with Time-Series Features

This project predicts gas_sales using historical car sales and oil price data. Key points:

Features:

ev_sales, hybrid_sales, oil_price and their interactions.

Lag features (1–4 previous periods) to capture temporal trends.

Rolling averages (3- and 5-period) to smooth short-term fluctuations.

Models:

Linear Regression

Random Forest Regressor

XGBoost Regressor

Results:

Adding lag and rolling features dramatically improved performance.

Linear Regression achieved R² ≈ 0.998, showing strong predictability from recent trends.

Feature importance highlights gas_sales_roll3, oil_x_ev, and short-term lags as the most influential predictors.

Conclusion:

Temporal features (history and trends) are critical for predicting time-dependent variables.

The pipeline can be adapted for other time-series regression tasks.
