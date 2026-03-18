# -------------------------------------------------------
# PHASE 3 : MACHINE LEARNING MODELS (RF + XGBoost)
# Project: Predictive Modelling of Drug Consumption Pattern
# -------------------------------------------------------

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

# -------------------------------------------------------
# Load preprocessed data
# -------------------------------------------------------
df = pd.read_csv("pharma_preprocessed.csv")

# Encode categorical columns
categorical_cols = df.select_dtypes(include=['object']).columns
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])

# Feature-target split
X = df.drop(columns=['sales'])
y = df['sales']

# Normalize numerical features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Save scaler for reuse in hybrid or DL models
joblib.dump(scaler, "ml_scaler.pkl")

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# -------------------------------------------------------
# Random Forest Model
# -------------------------------------------------------
rf = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

# Evaluate RF
mae_rf = mean_absolute_error(y_test, y_pred_rf)
mse_rf = mean_squared_error(y_test, y_pred_rf)
rmse_rf = np.sqrt(mse_rf)
r2_rf = r2_score(y_test, y_pred_rf)

print("\n--- Random Forest Metrics ---")
print(f"MAE: {mae_rf:.4f}")
print(f"MSE: {mse_rf:.4f}")
print(f"RMSE: {rmse_rf:.4f}")
print(f"R2 Score: {r2_rf:.4f}")

# Save RF model & predictions
joblib.dump(rf, "random_forest_model.pkl")
np.save("rf_predictions.npy", y_pred_rf)
np.save("y_test.npy", y_test)

# -------------------------------------------------------
# XGBoost Model
# -------------------------------------------------------
xgb = XGBRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1
)
xgb.fit(X_train, y_train)
y_pred_xgb = xgb.predict(X_test)

# Evaluate XGBoost
mae_xgb = mean_absolute_error(y_test, y_pred_xgb)
mse_xgb = mean_squared_error(y_test, y_pred_xgb)
rmse_xgb = np.sqrt(mse_xgb)
r2_xgb = r2_score(y_test, y_pred_xgb)

print("\n--- XGBoost Metrics ---")
print(f"MAE: {mae_xgb:.4f}")
print(f"MSE: {mse_xgb:.4f}")
print(f"RMSE: {rmse_xgb:.4f}")
print(f"R2 Score: {r2_xgb:.4f}")

# Save XGBoost model & predictions
joblib.dump(xgb, "xgboost_model.pkl")
np.save("xgb_predictions.npy", y_pred_xgb)

# -------------------------------------------------------
print("\n✅ Optimized ML Models Complete & Saved!")
print("Files saved:")
print(" - random_forest_model.pkl")
print(" - xgboost_model.pkl")
print(" - rf_predictions.npy, xgb_predictions.npy, y_test.npy")
