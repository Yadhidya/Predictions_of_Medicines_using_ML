import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

# -------------------------------------------------------
# Load Data
# -------------------------------------------------------
df = pd.read_csv("pharma_preprocessed.csv")

# Keep original for seasonal classification
df_original = df.copy()

# Encode categorical columns
categorical_cols = df.select_dtypes(include=['object']).columns
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])

# Features & Target
X = df.drop(columns=['sales'])
y = df['sales']

# Scale
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# -------------------------------------------------------
# Models
# -------------------------------------------------------
models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(max_depth=10, random_state=42),
    "Random Forest": RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1),
    "XGBoost": XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1
    )
}

results = []
predictions = {}

# -------------------------------------------------------
# Train & Evaluate Models
# -------------------------------------------------------
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    # MAPE (avoid division by zero)
    mape = np.mean(np.abs((y_test - y_pred) / (y_test + 1e-6))) * 100

    r2 = r2_score(y_test, y_pred)

    results.append([name, mae, rmse, mape, r2])
    predictions[name] = y_pred

# Convert to DataFrame
results_df = pd.DataFrame(results, columns=["Model", "MAE", "RMSE", "MAPE (%)", "R2"])

print("\n================ MODEL PERFORMANCE ================\n")
print(results_df)

# -------------------------------------------------------
# SEASONAL vs NON-SEASONAL PERFORMANCE
# -------------------------------------------------------

seasonal_classes = ["Antimalarial", "Antibiotics", "Antipiretics"]

df_test_original = df_original.iloc[y_test.index].copy()

df_test_original["category_type"] = df_test_original["product_class"].apply(
    lambda x: "Seasonal" if x in seasonal_classes else "Non-Seasonal"
)

# Best model selection
best_model_name = results_df.sort_values("R2", ascending=False)["Model"].iloc[0]
best_pred = predictions[best_model_name]

seasonal_results = []

for cat in ["Seasonal", "Non-Seasonal"]:
    mask = df_test_original["category_type"] == cat

    y_true = y_test[mask]
    y_p = best_pred[mask]

    mae = mean_absolute_error(y_true, y_p)
    rmse = np.sqrt(mean_squared_error(y_true, y_p))
    mape = np.mean(np.abs((y_true - y_p) / (y_true + 1e-6))) * 100
    r2 = r2_score(y_true, y_p)

    seasonal_results.append([cat, mae, rmse, mape, r2])

seasonal_df = pd.DataFrame(
    seasonal_results,
    columns=["Category", "MAE", "RMSE", "MAPE (%)", "R2"]
)

print("\n=========== SEASONAL vs NON-SEASONAL ===========\n")
print(seasonal_df)