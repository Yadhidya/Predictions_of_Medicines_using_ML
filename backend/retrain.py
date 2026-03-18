import pandas as pd
import numpy as np
import os
import joblib
import warnings

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import HistGradientBoostingRegressor
from category_encoders.target_encoder import TargetEncoder

warnings.filterwarnings("ignore")

DATA_PATH = "pharma_preprocessed.csv"
MODEL_DIR = "monthly_pc_model_fast"
os.makedirs(MODEL_DIR, exist_ok=True)

print("⚙️ Loading dataset...")
df = pd.read_csv(DATA_PATH)

# -------------------------------------------------
# Clean
# -------------------------------------------------
df = df.drop(
    columns=["date", "name_of_sales_rep", "latitude", "longitude", "product_name"],
    errors="ignore"
)

month_map = {
    "January":1,"February":2,"March":3,"April":4,
    "May":5,"June":6,"July":7,"August":8,
    "September":9,"October":10,"November":11,"December":12
}
df["month"] = df["month"].replace(month_map)
df["month"] = pd.to_numeric(df["month"], errors="coerce")
df = df.dropna(subset=["month"])
df["month"] = df["month"].astype(int)

print("🔹 Aggregating monthly product-class data...")

df_monthly = (
    df.groupby(
        ["country","city","product_class","month","year",
         "season","disease_category","city_class_key"],
        as_index=False
    )["sales"].sum()
)

print("Rows after aggregation:", len(df_monthly))

# -------------------------------------------------
# Features / Target
# -------------------------------------------------
y = df_monthly["sales"]
X = df_monthly.drop(columns=["sales"])

categorical_cols = X.select_dtypes(include="object").columns.tolist()
numeric_cols = X.select_dtypes(include=np.number).columns.tolist()

high_card_cols = ["city"]
low_card_cols = [c for c in categorical_cols if c not in high_card_cols]

# -------------------------------------------------
# Preprocessing
# -------------------------------------------------
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_cols),
        ("low_cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), low_card_cols),
        ("high_cat", TargetEncoder(), high_card_cols),
    ]
)

# -------------------------------------------------
# FAST MODEL
# -------------------------------------------------
model = HistGradientBoostingRegressor(
    max_depth=8,
    learning_rate=0.08,
    max_iter=250,
    random_state=42
)

pipeline = Pipeline([
    ("prep", preprocessor),
    ("model", model)
])

# -------------------------------------------------
# Train
# -------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("🚀 Training FAST model...")
pipeline.fit(X_train, y_train)

# -------------------------------------------------
# Evaluate
# -------------------------------------------------
pred = pipeline.predict(X_test)

print("\n✅ Model Performance")
print("MAE :", round(mean_absolute_error(y_test, pred), 2))
print("RMSE:", round(np.sqrt(mean_squared_error(y_test, pred)), 2))
print("R²  :", round(r2_score(y_test, pred), 4))

# -------------------------------------------------
# Save
# -------------------------------------------------
joblib.dump(pipeline, f"{MODEL_DIR}/hgb_monthly_pc_pipeline.joblib")
joblib.dump(
    {"feature_columns": X.columns.tolist()},
    f"{MODEL_DIR}/hgb_monthly_pc_metadata.joblib"
)

print("\n💾 FAST model saved safely")
print("🛡️ Training completed in minutes, not hours")
