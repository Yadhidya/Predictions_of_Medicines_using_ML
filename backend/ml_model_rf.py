# ml_model_rf.py
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import os
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

os.makedirs("rf_models", exist_ok=True)

df = pd.read_csv("pharma_preprocessed.csv")

drop_cols = ["date", "name_of_sales_rep", "latitude", "longitude"]
df = df.drop(columns=drop_cols, errors="ignore")

month_map = {
    'January': 1, 'February': 2, 'March': 3, 'April': 4,
    'May': 5, 'June': 6, 'July': 7, 'August': 8,
    'September': 9, 'October': 10, 'November': 11, 'December': 12
}

df['month'] = df['month'].replace(month_map)
df['month'] = pd.to_numeric(df['month'], errors='coerce')
df = df.dropna(subset=['month'])
df['month'] = df['month'].astype(int)

def get_season(month):
    if month in [12, 1, 2]: return 'Winter'
    elif month in [3, 4, 5]: return 'Spring'
    elif month in [6, 7, 8]: return 'Summer'
    else: return 'Autumn'

if 'season' not in df.columns:
    df['season'] = df['month'].apply(get_season)

df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

y = df['sales']
X = df.drop(columns=['sales'])

categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()

preprocessor = ColumnTransformer(transformers=[
    ("num", StandardScaler(), numeric_cols),
    ("cat", OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols)
])

rf = RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)

pipeline = Pipeline([
    ("preproc", preprocessor),
    ("rf", rf)
])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

pipeline.fit(X_train, y_train)

y_pred = pipeline.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"✅ RF pipeline results → MAE: {mae:.4f}, RMSE: {rmse:.4f}, R²: {r2:.4f}")

joblib.dump(pipeline, "rf_models/rf_pipeline.joblib")
joblib.dump({
    "categorical": categorical_cols,
    "numeric": numeric_cols,
    "feature_columns": X.columns.tolist()
}, "rf_models/rf_metadata.joblib")

print("✅ Model & metadata saved successfully in 'rf_models/' folder.")
