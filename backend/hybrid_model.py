import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import joblib
import os

# --- Load dataset ---
df = pd.read_csv("pharma_preprocessed.csv")

# Drop high-variance or non-predictive columns
drop_cols = ["date", "name_of_sales_rep", "latitude", "longitude"]
df = df.drop(columns=drop_cols, errors="ignore")

# --- Define target and features ---
y = df["sales"]
X = df.drop(columns=["sales"])

# --- Encode categorical features ---
encoders = {}
for col in X.select_dtypes(include=["object"]).columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))
    encoders[col] = le

# --- Scale numeric features ---
scaler = StandardScaler()
numeric_cols = X.select_dtypes(include=[np.number]).columns
X[numeric_cols] = scaler.fit_transform(X[numeric_cols])

# --- Split data for training ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# --- Random Forest Model ---
rf = RandomForestRegressor(n_estimators=300, random_state=42)
rf.fit(X_train, y_train)
rf_preds = rf.predict(X_test)

# --- Deep Neural Network ---
dnn = Sequential([
    Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(64, activation='relu'),
    Dense(1)
])
dnn.compile(optimizer='adam', loss='mse')
dnn.fit(X_train, y_train, epochs=30, batch_size=32, verbose=0)
dnn_preds = dnn.predict(X_test).flatten()

# --- Hybrid Prediction (average of both) ---
hybrid_preds = (rf_preds + dnn_preds) / 2

# --- Evaluation ---
mae = mean_absolute_error(y_test, hybrid_preds)
mse = mean_squared_error(y_test, hybrid_preds)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, hybrid_preds)

print("\n--- Hybrid Model Metrics ---")
print(f"MAE: {mae:.4f}")
print(f"MSE: {mse:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R² Score: {r2:.4f}")

# --- Save models & encoders ---
os.makedirs("hybrid_models", exist_ok=True)
joblib.dump(rf, "hybrid_models/random_forest.pkl")
dnn.save("hybrid_models/dnn_model.keras")
joblib.dump(encoders, "hybrid_models/encoders.pkl")
joblib.dump(scaler, "hybrid_models/scaler.pkl")
joblib.dump(list(X.columns), "hybrid_models/feature_columns.pkl")

print("\n✅ Hybrid model training complete and saved in hybrid_models/")
