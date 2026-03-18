import pandas as pd
import numpy as np
import os, joblib, warnings

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.callbacks import EarlyStopping

warnings.filterwarnings("ignore")
DATA_PATH = "pharma_preprocessed.csv"
MODEL_DIR = "dl_models"
os.makedirs(MODEL_DIR, exist_ok=True)

print("Loading dataset...")
df = pd.read_csv(DATA_PATH)

month_map = {
    "January":1,"February":2,"March":3,"April":4,
    "May":5,"June":6,"July":7,"August":8,
    "September":9,"October":10,"November":11,"December":12
}
df["month"] = df["month"].replace(month_map)
df["month"] = pd.to_numeric(df["month"], errors="coerce")
df = df.dropna(subset=["month"])
df["month"] = df["month"].astype(int)

def get_season(m):
    if m in [12,1,2]: return "Winter"
    if m in [3,4,5]: return "Spring"
    if m in [6,7,8]: return "Summer"
    return "Autumn"

if "season" not in df.columns:
    df["season"] = df["month"].apply(get_season)

DROP_COLS = ["date", "name_of_sales_rep", "latitude", "longitude"]
df = df.drop(columns=DROP_COLS, errors="ignore")

categorical_cols = df.select_dtypes(include="object").columns.tolist()
encoders = {}

for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le

joblib.dump(encoders, f"{MODEL_DIR}/label_encoders.pkl")

X = df.drop(columns=["sales"])
y = df["sales"].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
joblib.dump(scaler, f"{MODEL_DIR}/scaler.pkl")

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)
print("\nTraining LSTM ")

X_train_lstm = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
X_test_lstm  = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))

lstm = Sequential([
    LSTM(64, return_sequences=False, input_shape=(1, X_train.shape[1])),
    Dropout(0.2),
    Dense(32, activation="relu"),
    Dense(1)
])

lstm.compile(optimizer="adam", loss="mse", metrics=["mae"])

lstm.fit(
    X_train_lstm, y_train,
    validation_split=0.2,
    epochs=60,
    batch_size=64,
    callbacks=[EarlyStopping(patience=8, restore_best_weights=True)],
    verbose=1
)

lstm_pred = lstm.predict(X_test_lstm).flatten()

print("\ LSTM Performance")
print("MAE :", round(mean_absolute_error(y_test, lstm_pred), 2))
print("RMSE:", round(np.sqrt(mean_squared_error(y_test, lstm_pred)), 2))
print("R²  :", round(r2_score(y_test, lstm_pred), 4))

lstm.save(f"{MODEL_DIR}/lstm_model.h5")

np.save(f"{MODEL_DIR}/lstm_predictions.npy", lstm_pred)
np.save(f"{MODEL_DIR}/y_test.npy", y_test)

print("\n Models saved in dl_models/")
